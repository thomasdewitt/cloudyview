"""What the volume texture's boundary does, now that it has no border.

Until 2026-08-15 both hosts uploaded the field ghost-padded by one texel per
side: zeros, so that hardware trilinear filtering tapered out of the field
instead of smearing the edge voxel outward, or — in a doubly periodic domain —
the opposite faces, so that filtering across the wrap seam was exact. It cost
two texels on every axis, which is why a 2048-cell field did not open in a
browser reporting the WebGPU spec floor of 2048 and a 2046-cell one did.

Both behaviours are now computed instead of uploaded (raymarch.wgsl
sample_level): the wrap is a repeat address mode on the two lateral texture
axes, and the taper is an analytic window multiplying a clamp-to-edge fetch.
Neither is an approximation of the old scheme — they are algebraically the
same function — and this file is where that claim is checked rather than
asserted.

Two levels of check, because they fail in different ways:

  * the identity, in float64 numpy, against an explicit model of the padded
    texture the renderer used to build. No GPU. This is the algebra.
  * the shader's own sample_level, lifted verbatim out of raymarch.wgsl and
    run on a real device against the same model. This is the algebra actually
    reaching the card: the coordinate convention, the axis swizzle, and which
    sampler each branch uses. Skips without a GPU.

This replaces tests/test_soar_texture_parity.py, which byte-diffed the two
hosts' ghost rings against each other. That test existed because the browser
wrapped the ring and the Python host shipped zeros there for the entire life
of the periodic renderer, silently, including in all eight goldens. There is
no ring left for the two hosts to disagree about — they both upload the bare
field — so the risk it guarded is gone by construction, and what is worth
guarding instead is the arithmetic that took over.
"""

import numpy as np
import pytest

REPO_SHADER = "web/soar/raymarch.wgsl"


# --- models -----------------------------------------------------------------

def trilinear_clamp(vol, coord):
    """One texel-space trilinear fetch with clamp-to-edge, in float64.

    `coord` is in texel units: value at index i sits at coord i. This is what
    a filtering sampler does, minus the hardware's sub-texel weight
    quantization.
    """
    lo = np.floor(coord).astype(int)
    w = coord - lo
    total = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = ((w[0] if dx else 1.0 - w[0])
                          * (w[1] if dy else 1.0 - w[1])
                          * (w[2] if dz else 1.0 - w[2]))
                if weight == 0.0:
                    continue
                i = np.clip(lo + np.array([dx, dy, dz]), 0,
                            np.array(vol.shape) - 1)
                total += weight * vol[tuple(i)]
    return total


def ghost_padded(field, periodic):
    """The texture soar used to upload: (nx+2, ny+2, nz+2), voxel i at i+1.

    Verbatim the old cloudyview.soar_host.write_wrap_ghosts placement, kept
    here rather than imported because it no longer exists in the renderer and
    this file is the only thing that still needs to know what it did. The four
    corner columns wrap in BOTH x and y — they are the trilinear support of a
    sample near a domain corner — and z never wraps.
    """
    padded = np.zeros(np.array(field.shape) + 2, np.float64)
    padded[1:-1, 1:-1, 1:-1] = field
    if periodic:
        core = padded[1:-1, 1:-1, 1:-1]
        padded[0, 1:-1, 1:-1] = core[-1]
        padded[-1, 1:-1, 1:-1] = core[0]
        padded[1:-1, 0, 1:-1] = core[:, -1]
        padded[1:-1, -1, 1:-1] = core[:, 0]
        padded[0, 0, 1:-1] = core[-1, -1]
        padded[0, -1, 1:-1] = core[-1, 0]
        padded[-1, 0, 1:-1] = core[0, -1]
        padded[-1, -1, 1:-1] = core[0, 0]
    return padded


def old_sample(field, g, periodic):
    """What the renderer returned before de-padding: a fetch at g + 1."""
    return trilinear_clamp(ghost_padded(field, periodic),
                           np.asarray(g, np.float64) + 1.0)


def edge_taper(g, n):
    """raymarch.wgsl edge_taper, in numpy."""
    return float(np.clip(min(g + 1.0, n - g), 0.0, 1.0))


def new_sample(field, g, periodic):
    """What sample_level computes now: one fetch, times the window.

    The periodic branch wraps laterally in the sampler (modelled here by
    taking the texel index modulo n) and tapers only in z; everything else
    clamps and tapers on all three axes.
    """
    g = np.asarray(g, np.float64)
    n = np.array(field.shape, np.float64)
    if periodic:
        lo = np.floor(g).astype(int)
        w = g - lo
        total = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight = ((w[0] if dx else 1.0 - w[0])
                              * (w[1] if dy else 1.0 - w[1])
                              * (w[2] if dz else 1.0 - w[2]))
                    if weight == 0.0:
                        continue
                    ix = (lo[0] + dx) % field.shape[0]
                    iy = (lo[1] + dy) % field.shape[1]
                    iz = min(max(lo[2] + dz, 0), field.shape[2] - 1)
                    total += weight * field[ix, iy, iz]
        return total * edge_taper(g[2], n[2])
    taper = (edge_taper(g[0], n[0]) * edge_taper(g[1], n[1])
             * edge_taper(g[2], n[2]))
    return trilinear_clamp(field, g) * taper


# --- the field and the query set --------------------------------------------

def synthetic_field(nx=5, ny=4, nz=3):
    """Small, and nonzero on every face, edge and corner.

    A wrong face, a transposed axis or a missing corner all have to show, so
    no two lateral faces carry the same value and no corner column matches the
    faces it sits on.
    """
    rng = np.random.default_rng(20260815)
    field = 0.1 + 0.8 * rng.random((nx, ny, nz))
    field[0, :, :] = 0.125
    field[-1, :, :] = 0.25
    field[:, 0, :] = 0.5
    field[:, -1, :] = 0.75
    field[0, 0, :] = 1.5
    field[0, -1, :] = 1.75
    field[-1, 0, :] = 2.5
    field[-1, -1, :] = 3.5
    # fp16 up front: the texture is r16float, so both sides must start from
    # values it represents exactly or the comparison measures rounding.
    return np.asarray(np.asarray(field, np.float16), np.float64)


def query_points(field, periodic):
    """Data coordinates worth asking about, edge to edge and past it.

    The interior is the easy case and the least interesting. What matters is
    the outer half-cell on each side (where the taper and the wrap live), the
    corners (where two or three of them act at once), and coordinates outside
    the box entirely, which gradient taps reach on every frame.

    Laterally, the periodic set stops one cell outside the domain, because
    that is as far as the old ghost ring was DEFINED: it was one texel deep,
    so past it the padded texture clamped and stopped tiling. The unpadded
    sampler keeps tiling — see test_the_wrap_keeps_tiling_past_the_old_ring —
    which is both more correct and unreachable, since wrap_to_domain folds
    every sample point into the domain before any of this runs. The vertical
    needs no such limit: both schemes are flatly zero beyond one cell out.
    """
    nx, ny, nz = (float(n) for n in field.shape)
    rng = np.random.default_rng(9012)
    lateral = lambda n: [m for m in
                         [-2.0, -1.0, -0.7, -0.5, -0.25, 0.0, 0.3, 0.5,
                          n / 2.0, n - 1.5, n - 1.0, n - 0.6, n - 0.5,
                          n - 0.25, n, n + 0.25, n + 1.5]
                         if not periodic or -1.0 <= m <= n]
    pts = []
    for gx in lateral(nx):
        for gy in lateral(ny):
            for gz in (-0.5, 0.0, nz / 2.0, nz - 1.0, nz - 0.5, nz + 0.5):
                pts.append([gx, gy, gz])
    # Random, to catch anything the grid happens to sit exactly on.
    span = np.array([nx, ny, nz]) + 3.0
    extra = rng.random((400, 3)) * span - 1.5
    if periodic:
        extra[:, 0] = np.clip(extra[:, 0], -1.0, nx)
        extra[:, 1] = np.clip(extra[:, 1], -1.0, ny)
    pts.extend(extra.tolist())
    return np.asarray(pts, np.float64)


# --- the identity, without a GPU --------------------------------------------

@pytest.mark.parametrize("periodic", [False, True])
def test_analytic_taper_reproduces_the_ghost_ring(periodic):
    """The claim the de-padding rests on, at every coordinate that matters.

    Trilinear filtering is separable and the old border was zero (or the far
    face) on every axis, so the window is an algebraic identity rather than a
    fit — the two sides differ only in the order the same products are
    summed, which in float64 costs a few ULPs (measured worst: 2e-15). The
    gate is three orders under anything a real mistake could produce and ten
    orders over that: an off-by-a-half-texel moves a sample by order 1 here.
    """
    field = synthetic_field()
    worst = 0.0
    for g in query_points(field, periodic):
        old = old_sample(field, g, periodic)
        new = new_sample(field, g, periodic)
        worst = max(worst, abs(old - new))
    assert worst < 1e-12, (
        f"the unpadded sample departs from the ghost-padded one by {worst:g} "
        f"(periodic={periodic}); the analytic taper is not the same function "
        "as the border it replaced")


def test_the_taper_is_one_across_the_whole_interior():
    """No window inside the data. A taper that bit at, say, g = 0.5 would dim
    the outermost half-cell of every field and look like a plausible
    boundary."""
    field = synthetic_field()
    for axis, n in enumerate(field.shape):
        for g in np.linspace(0.0, n - 1.0, 37):
            assert edge_taper(float(g), float(n)) == 1.0, (
                f"axis {axis} tapers at data coordinate {g} of {n}")


def test_the_wrap_keeps_tiling_past_the_old_ring():
    """The one place the two schemes deliberately differ, stated on purpose.

    A one-texel ghost ring can only carry one cell of the far side; past that
    the padded texture clamped, so a periodic field stopped tiling one cell
    out. The repeat sampler does not — g and g + n_cells return the same
    value however far out you go. Nothing reaches that range (wrap_to_domain
    folds every sample point into the domain before sampling, and the
    gradient taps go through the same fold), and where it did, tiling is the
    right answer for a periodic domain. Pinned rather than left implicit,
    because it is the one difference the identity test above cannot cover.
    """
    field = synthetic_field()
    nx, ny, _ = field.shape
    for g in ([-3.4, 1.2, 1.0], [nx + 2.5, -5.5, 0.4], [2.5, ny + 3.5, 1.7]):
        tiled = [g[0] + nx, g[1] + ny, g[2]]
        assert new_sample(field, g, True) == pytest.approx(
            new_sample(field, tiled, True), abs=1e-12)


def test_a_periodic_seam_carries_the_far_face():
    """Half a cell outside x = 0 in a periodic domain is the mean of the two
    faces — not a fade to zero, which is what a clamp without the wrap would
    give and what the goldens tapered into for a year."""
    field = synthetic_field()
    nx, ny, nz = field.shape
    for iy in range(ny):
        for iz in range(nz):
            got = new_sample(field, [-0.5, float(iy), float(iz)], True)
            want = 0.5 * (field[0, iy, iz] + field[nx - 1, iy, iz])
            assert got == pytest.approx(want, abs=1e-12)


# --- the same arithmetic, on a real device ----------------------------------

def _extract_wgsl_function(source, name):
    """Lift one `fn name(...) {...}` out of the shader by brace matching.

    Verbatim, so this test pins the code the browser runs rather than a copy
    of it that can drift. A rename must fail loudly here.
    """
    start = source.find(f"fn {name}(")
    if start < 0:
        raise AssertionError(
            f"raymarch.wgsl has no `fn {name}(` — the shader and this test "
            "have drifted apart.")
    depth = 0
    i = source.index("{", start)
    for j in range(i, len(source)):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[start:j + 1]
    raise AssertionError(f"unbalanced braces in `fn {name}`")


HARNESS = """
@group(0) @binding(0) var vol: texture_3d<f32>;
@group(0) @binding(1) var vol_samp: sampler;
@group(0) @binding(2) var vol_wrap_samp: sampler;
@group(0) @binding(3) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> results: array<f32>;

%s

%s

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&queries)) { return; }
    // A deliberately non-trivial AABB (matching AABB_BMIN / AABB_VOXEL on
    // the Python side), so the world-to-data mapping is under test too:
    // cell centre i sits at box fraction (i + 0.5) / N, and the probe's
    // world points only land back on integer data coordinates if
    // sample_level applies exactly that convention.
    let td = vec3<f32>(textureDimensions(vol, 0));
    let dims = vec3<f32>(td.z, td.y, td.x);
    let vox = vec3<f32>(12.5, 3.0, 90.0);
    let bmin = vec3<f32>(100.0, -40.0, 7.0);
    results[i] = sample_level(vol, queries[i].xyz, bmin, bmin + dims * vox,
                              queries[i].w > 0.5);
}
"""

# Must match `vox` and `bmin` in HARNESS above.
AABB_VOXEL = np.array([12.5, 3.0, 90.0])
AABB_BMIN = np.array([100.0, -40.0, 7.0])


@pytest.fixture(scope="module")
def gpu_sampler_probe():
    """sample_level, compiled from the real shader onto the real device."""
    wgpu = pytest.importorskip("wgpu")
    from pathlib import Path

    try:
        from .conftest import soar_gpu_adapter
    except ImportError:
        from conftest import soar_gpu_adapter

    adapter = soar_gpu_adapter()
    if adapter is None:
        pytest.skip("no usable GPU adapter (a software rasterizer is not one)")
    device = adapter.request_device_sync()

    source = (Path(__file__).resolve().parents[1] / REPO_SHADER).read_text()
    code = HARNESS % (_extract_wgsl_function(source, "edge_taper"),
                      _extract_wgsl_function(source, "sample_level"))
    module = device.create_shader_module(code=code)

    def probe(field, points, periodic):
        # Queries arrive in data coordinates; the buffer carries world
        # points inside the harness AABB. Cell centre g sits at world
        # bmin + (g + 0.5) * voxel — the half-cell-padded convention.
        points = AABB_BMIN + (np.asarray(points, np.float64) + 0.5) * AABB_VOXEL
        data = np.ascontiguousarray(field, np.float16)
        nx, ny, nz = data.shape
        texture = device.create_texture(
            size=(nz, ny, nx), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture(
            {"texture": texture}, data.tobytes(),
            {"bytes_per_row": nz * 2, "rows_per_image": ny}, (nz, ny, nx))

        # The two samplers soar_host.SoarRenderer builds, restated so a change
        # to either address mode shows up here as a wrong number.
        clamp = device.create_sampler(
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge", mag_filter="linear",
            min_filter="linear")
        wrap = device.create_sampler(
            address_mode_u="clamp-to-edge", address_mode_v="repeat",
            address_mode_w="repeat", mag_filter="linear", min_filter="linear")

        query = np.zeros((len(points), 4), np.float32)
        query[:, :3] = points
        query[:, 3] = 1.0 if periodic else 0.0
        qbuf = device.create_buffer_with_data(
            data=query.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        nbytes = 4 * len(points)
        rbuf = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        staging = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)

        layout = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": "float", "view_dimension": "3d"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "sampler": {"type": "filtering"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "sampler": {"type": "filtering"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "read-only-storage"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "storage"}},
        ])
        pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={"module": module, "entry_point": "main"})
        bind_group = device.create_bind_group(layout=layout, entries=[
            {"binding": 0, "resource": texture.create_view()},
            {"binding": 1, "resource": clamp},
            {"binding": 2, "resource": wrap},
            {"binding": 3, "resource": {"buffer": qbuf, "offset": 0,
                                        "size": query.nbytes}},
            {"binding": 4, "resource": {"buffer": rbuf, "offset": 0,
                                        "size": nbytes}},
        ])
        encoder = device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups((len(points) + 63) // 64)
        pass_.end()
        encoder.copy_buffer_to_buffer(rbuf, 0, staging, 0, nbytes)
        device.queue.submit([encoder.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        out = np.frombuffer(bytearray(staging.read_mapped()),
                            np.float32).astype(np.float64)
        staging.unmap()
        texture.destroy()
        return out

    return probe


# Hardware filtering interpolates with quantized sub-texel weights (8 bits of
# fraction is typical), so the device cannot be held to the float64 identity
# above. Measured on an RTX 5080 (Vulkan) over the query set: worst 0.008
# non-periodic and 0.011 periodic, mean 1e-4, which is what a 1/256 weight
# step costs across the ~3-wide contrast this field carries at its seams.
#
# The gate sits at twice that and three orders under what a structural
# mistake costs: an off-by-one texel, a missing wrap, or a taper on the wrong
# axis all move a sample by an appreciable fraction of that same contrast.
GPU_SAMPLE_TOLERANCE = 0.02


@pytest.mark.parametrize("periodic", [False, True])
def test_gpu_sample_level_matches_the_ghost_padded_model(gpu_sampler_probe,
                                                         periodic):
    """The shader's own function, on the card, against the old scheme."""
    field = synthetic_field()
    points = query_points(field, periodic)
    got = gpu_sampler_probe(field, points, periodic)
    want = np.array([old_sample(field, g, periodic) for g in points])
    worst = int(np.argmax(np.abs(got - want)))
    assert np.allclose(got, want, atol=GPU_SAMPLE_TOLERANCE), (
        f"sample_level disagrees with the ghost-padded model "
        f"(periodic={periodic}): worst at data coordinate "
        f"{points[worst].tolist()}, GPU {got[worst]:.6f} vs model "
        f"{want[worst]:.6f}")


def test_a_2048_cell_field_fits_a_2048_texel_limit():
    """The thing all of the above is for.

    Chrome's Dawn reports maxTextureDimension3D = 2048 whatever the card can
    do, and LES and STEAM fields come out 2048 cells across; 2046 is not a
    number anybody simulates. Ghost-padded, that field asked for 2050 and was
    refused. Held to the same limit here — the device is created with the
    spec floor requested explicitly, so this runs on hardware that reports
    16384 — it now fits exactly, and 2049 still fails with a sentence.
    """
    wgpu = pytest.importorskip("wgpu")
    try:
        from .conftest import soar_gpu_adapter
    except ImportError:
        from conftest import soar_gpu_adapter
    from cloudyview.soar_host import SoarRenderer

    adapter = soar_gpu_adapter()
    if adapter is None:
        pytest.skip("no usable GPU adapter (a software rasterizer is not one)")
    device = adapter.request_device_sync(
        required_limits={"max-texture-dimension-3d": 2048})
    assert device.limits.get("max-texture-dimension-3d") == 2048

    renderer = SoarRenderer(device=device, periodic=True, nested=False)
    # Thin in z on purpose: this is about the lateral axes, and 2048x2048x4
    # fp16 is 33 MB rather than the gigabytes a realistic depth would cost.
    renderer.upload_volume(np.zeros((2048, 2048, 4), np.float16))
    assert renderer._vol_tex.size == (4, 2048, 2048)

    with pytest.raises(ValueError, match="2049"):
        renderer.upload_volume(np.zeros((2049, 16, 4), np.float16))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
