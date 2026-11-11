"""SoarRenderer: the shader on a GPU, through wgpu.

One renderer holds a device, the shader module of one surface (shader.py),
that surface's tile, and one uploaded field. render() draws a view of it:
each accumulated pass runs the surface stages (over the city), then the cloud
march, then a pass that folds the new sample into a running mean. The browser
app runs the same shader files, so a picture made here is the picture soar
shows.

The volume texture is indexed width=z, height=y, depth=x, which means a
C-order (nx, ny, nz) numpy array is already in the right byte order. Do not
transpose.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional, Tuple

import numpy as np

from . import shader
from .tiers import LIGHT_CACHE_DIVISOR, QUALITY_PRESETS
from .surfaces import (SURFACE_DIRS, SURFACE_FORMAT_CHANNELS, SURFACE_STAGES,
                       SURFACE_TILE_SCHEMAS, VEIL_STAGES, check_surface,
                       decode_surface_level, surface_level_files,
                       surface_stage_plan)
from .uniforms import (SLOT_BAKE_SLICE, SLOT_LENS_DT, UNIFORM_NBYTES,
                       SceneState, ViewState, pack_uniforms)

# Passes render() accumulates when the caller does not say.
STILL_ACCUMULATE_FRAMES = 64
# How long one submitted band of a pass should take, in seconds, and how many
# pixels the first band of a frame covers before anything is known about the
# scene's cost. See SoarRenderer.render: macOS's GPU watchdog kills a command
# buffer that runs too long.
SUBMIT_BUDGET_S = 0.25
INITIAL_BAND_PIXELS = 65536

# The march's own output for one pass. Half precision is plenty for a single
# sample: its round-off is ~0.03 8-bit levels, unbiased, and it averages away.
SAMPLE_FORMAT = "rgba16float"
# The baked light cache: optical depth to the sun and optical depth overhead.
LIGHT_CACHE_FORMAT = "rg16float"
# The running mean. It is a feedback loop: this pass's output is the next
# pass's input, so its round-off does not average away, it accumulates. In
# half precision the mean drifts darker with the number of passes (~10 8-bit
# levels from 1 to 1024 passes). Full precision removes that.
ACCUM_FORMAT = "rgba32float"
# The surface stages' target: (radiance, t) of the traced city under each
# pixel. t runs to 1e30 on a miss and to tens of km on a hit, so it needs
# float32. The march only ever textureLoad()s it.
SURFACE_FORMAT = "rgba32float"
# The veil that VEIL_STAGES write beside the surface (veil_pack in
# shader/passes.wgsl): half-float radiance and opacity and an f32 depth,
# packed as uints so that no float conversion touches the bits.
VEIL_FORMAT = "rgba32uint"

# Folds one pass's sample into the running mean:
# mean = prev * prev_weight + sample * sample_weight.
_ACCUM_SHADER = """
struct AccumUniforms { prev_weight: f32, sample_weight: f32, _p0: f32, _p1: f32 };
@group(0) @binding(0) var<uniform> au: AccumUniforms;
@group(0) @binding(1) var sample_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_tex: texture_2d<f32>;
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(frag_pos.xy);
    let s = textureLoad(sample_tex, xy, 0);
    if (au.prev_weight <= 0.0) { return vec4<f32>(s.rgb, 1.0); }
    let prev = textureLoad(prev_tex, xy, 0);
    return vec4<f32>(prev.rgb * au.prev_weight + s.rgb * au.sample_weight, 1.0);
}
"""

# Lens water, after accumulation: the accumulated image seen through the wet
# lens (lens_apply in shader/lens.wgsl), with the lens state that
# fs_lens_state (shader/passes.wgsl) wrote. The browser's present pass
# includes the same lens.wgsl; this is the same pass without the present's
# dither (image.quantize_uint8 dithers when a Python render is saved).
_LENS_POST_SHADER = """
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var lens_state_tex: texture_2d<f32>;
fn lens_src(uv: vec2<f32>) -> vec3<f32> { return lens_bilinear(src_tex, uv); }
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex, 0));
    let rgb = textureLoad(src_tex, vec2<i32>(frag_pos.xy), 0).rgb;
    let st = textureLoad(lens_state_tex, vec2<i32>(0, 0), 0);
    return vec4<f32>(lens_apply(rgb, frag_pos.xy / dims, dims.x / dims.y,
                                vec2<f32>(1.0) / dims, st), 1.0);
}
"""


class SoarRenderer:
    """A device, one surface's shader, and one uploaded field.

    Stateful and reusable on purpose: a video or a demo bake renders hundreds
    of frames of one field, and re-creating the device or re-uploading a
    400 MB volume per frame would cost more than the rendering.
    """

    def __init__(self, *, periodic: bool = True, nested: bool = False,
                 tone_map: bool = True, surface: str = "ocean", device=None,
                 shader_source: Optional[str] = None):
        """`surface` is what the world stands on: "ocean" or the night "city".
        Each has its own shader module and its own tile at binding 3, both
        fixed here, so the surface belongs to the renderer and not to a
        frame. The SceneState passed to each render must say the same
        surface, and the same `periodic` and `nested`.

        `tone_map=False` compiles the tone map out, and render() then returns
        linear radiance.

        `shader_source` replaces the shader files for `surface`
        (shader.assemble(surface), without the switch header). It is how
        tools/soar_shader_ab.py renders two revisions of the shader on one
        device to compare them.
        """
        import wgpu                                    # local: GPU only on use
        self.wgpu = wgpu
        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        self.device = device
        self.periodic = bool(periodic)
        self.nested = bool(nested)
        self.tone_map = bool(tone_map)
        self.surface = check_surface(surface)

        # One module per set of compile-time switches (see _spec), compiled
        # when a render first asks for it. The default lighting (the march
        # to the sun at every sample) is compiled here so that a shader that
        # does not compile fails at construction, not at the first frame.
        self._shader_source = (shader_source if shader_source is not None
                               else shader.assemble(self.surface))
        self._modules = {}
        self._pipelines = {}
        self._module_for(self._spec(light_cache=False))
        self._accum_module = device.create_shader_module(code=_ACCUM_SHADER)

        self._uniform_buf = device.create_buffer(
            size=UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._accum_buf = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._vol_sampler = device.create_sampler(
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge", mag_filter="linear", min_filter="linear")
        # The periodic read of the same volume. Texture axes: u indexes field
        # z (never periodic), v field y, w field x. Repeat on the two lateral
        # axes makes the filtering exact across the wrap seam (sample_level
        # in shader/volume.wgsl).
        self._vol_wrap_sampler = device.create_sampler(
            address_mode_u="clamp-to-edge", address_mode_v="repeat",
            address_mode_w="repeat", mag_filter="linear", min_filter="linear")
        self._ocean_sampler = device.create_sampler(
            address_mode_u="repeat", address_mode_v="repeat",
            mag_filter="linear", min_filter="linear", mipmap_filter="linear")

        self._vol_tex = None
        self._nest_tex = self._dummy_volume()
        # Binding 7 starts as the same 1x1x1 zero texture the nest uses;
        # bake_light_cache replaces it with a real cache, and render() calls
        # that whenever the cache is not the one for its scene and sun
        # (_light_cache_key says which it holds).
        self._light_tau_tex = self._dummy_volume()
        self._light_cache_key = None
        # Binding 8 is the browser's ice-detection mode. It is never read
        # here (u.ice_mode stays 0), but the layout requires it.
        self._ice_frac_tex = self._dummy_volume()
        # Group 1's stand-ins: what the march reads there over a surface
        # with no stages.
        self._surface_dummy = self.device.create_texture(
            size=(1, 1, 1), dimension="2d", format=SURFACE_FORMAT,
            usage=self.wgpu.TextureUsage.TEXTURE_BINDING)
        self._veil_dummy = self.device.create_texture(
            size=(1, 1, 1), dimension="2d", format=VEIL_FORMAT,
            usage=self.wgpu.TextureUsage.TEXTURE_BINDING)
        self._light_cache_dims = None
        # Binding 3 is the surface's tile (the ocean's normals, the city's
        # density).
        self._ocean_tex = self._load_surface_tile()
        self._targets = None
        self._bind_group = None
        self._band_rows = {}
        self._build_pipelines()

    # -- resources ---------------------------------------------------------

    def _dummy_volume(self):
        """A 1x1x1 volume of zero. Every binding of the layout must be bound
        whether or not the shader reads it (the nest, the light cache)."""
        wgpu = self.wgpu
        tex = self.device.create_texture(
            size=(1, 1, 1), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": tex}, np.zeros(1, np.float16).tobytes(),
            {"bytes_per_row": 2, "rows_per_image": 1}, (1, 1, 1))
        return tex

    def _load_surface_tile(self):
        """The tile bound at binding 3: pre-built mips, level by level.

        Which tile depends on the surface: the ocean's wave normals or the
        city's density field (both rgba16float). Each tile's meta.json names
        its format, files, encoding and compression. `self.surface_meta` is
        the loaded meta.json, where a caller reads the cell size and tile
        extent that SceneState has to be told (surfaces.surface_cell_m).
        """
        import gzip
        import json
        wgpu = self.wgpu
        tile_dir = SURFACE_DIRS[self.surface]
        meta = json.loads((tile_dir / "meta.json").read_text())
        expected = SURFACE_TILE_SCHEMAS[self.surface]
        if meta.get("schema") != expected:
            raise RuntimeError(
                f"{tile_dir}/meta.json is schema {meta.get('schema')!r}; this "
                f"renderer reads {expected!r}. Re-run "
                "tools/export_web_assets.py, or update the shader with it.")
        n, mips = int(meta["n"]), int(meta["mips"])
        fmt = meta.get("format", "rgba16float")
        texel_bytes = SURFACE_FORMAT_CHANNELS[fmt] * 2
        self.surface_meta = meta
        tex = self.device.create_texture(
            size=(n, n, 1), dimension="2d", format=fmt,
            mip_level_count=mips,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for level, name in enumerate(surface_level_files(meta)):
            side = max(1, n >> level)
            raw = (tile_dir / name).read_bytes()
            if meta.get("compression") == "gzip":
                raw = gzip.decompress(raw)
            try:
                data = decode_surface_level(raw, side, meta)
            except RuntimeError as exc:
                raise RuntimeError(f"{tile_dir.name}/{name}: {exc}") from None
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": level}, data,
                {"bytes_per_row": side * texel_bytes, "rows_per_image": side},
                (side, side, 1))
        return tex

    def upload_volume(self, sigma: np.ndarray) -> None:
        """Upload the extinction field.

        `sigma` is (nx, ny, nz) — the field itself, with no border of any
        kind. A C-order array of that shape is already z-fastest, which is
        exactly the byte order a texture of size (nz, ny, nx) wants, so there
        is no transpose here and adding one is a bug.

        There is nothing to pad and nothing to wrap. What the field does at
        its edges (a linear taper into zero, or the periodic lateral wrap) is
        computed by the shader (sample_level in shader/volume.wgsl) from the
        texels uploaded here, so a 2048-cell axis fits a 2048-texel limit.
        """
        wgpu = self.wgpu
        nx, ny, nz = sigma.shape
        limit = self.device.limits.get("max-texture-dimension-3d", 2048)
        if max(nx, ny, nz) > limit:
            raise ValueError(
                f"Volume {nx}x{ny}x{nz} exceeds this device's 3D texture "
                f"limit of {limit} on one axis.")
        if self._vol_tex is not None:
            self._vol_tex.destroy()
        self._vol_tex = self.device.create_texture(
            size=(nz, ny, nx), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._write_r16_volume(self._vol_tex, sigma)
        self._light_cache_key = None
        self._bind_group = None

    # Texels per write_texture call: the fp16 copy of the field is made a
    # slab at a time, never whole.
    _UPLOAD_SLAB_TEXELS = 1 << 24

    def _write_r16_volume(self, tex, sigma: np.ndarray) -> None:
        """Write (nx, ny, nz) `sigma` into r16float texture `tex` (nz, ny, nx).

        In slabs along x, the texture's depth, so the fp16 copy is a slab's
        and not the field's (a whole-field copy and its bytes are two fp16
        fields on top of the float one: 8.3 GB at 2048 x 2048 x 495). An
        x-slab of a C-order (nx, ny, nz) array is contiguous
        and z-fastest, exactly a run of the texture's depth slices, so the
        texels written are the same bytes in the same places as one whole
        write. queue.write_texture, never copy_buffer_to_texture: the 256-byte
        bytes_per_row rule applies to buffer copies, and nz*2 almost never
        satisfies it.
        """
        nx, ny, nz = sigma.shape
        step = max(1, self._UPLOAD_SLAB_TEXELS // max(1, ny * nz))
        for i0 in range(0, nx, step):
            slab = np.ascontiguousarray(sigma[i0:i0 + step], dtype=np.float16)
            self.device.queue.write_texture(
                {"texture": tex, "origin": (0, 0, i0)}, slab,
                {"bytes_per_row": nz * 2, "rows_per_image": ny},
                (nz, ny, slab.shape[0]))

    def upload_nest(self, sigma: np.ndarray) -> None:
        """Upload the finer level bound at binding 5.

        Always tapered to zero at its edges, even in a periodic domain: that
        taper is how the nest blends out into the coarse field around it,
        which is a different thing from the outer domain's wrap. The shader
        applies it; like the outer level, this is the bare (nx, ny, nz) field.
        """
        if not self.nested:
            raise RuntimeError(
                "This renderer was built with nested=False, so the shader has "
                "NESTED compiled to false and would ignore the nest.")
        wgpu = self.wgpu
        nx, ny, nz = sigma.shape
        if self._nest_tex is not None:
            self._nest_tex.destroy()
        self._nest_tex = self.device.create_texture(
            size=(nz, ny, nx), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._write_r16_volume(self._nest_tex, sigma)
        self._light_cache_key = None
        self._bind_group = None

    # -- pipelines ---------------------------------------------------------

    def _build_pipelines(self):
        wgpu = self.wgpu
        d = self.device
        tex3d = {"sample_type": "float", "view_dimension": "3d"}
        vis = wgpu.ShaderStage.FRAGMENT
        self._ray_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": vis,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": vis, "texture": tex3d},
            {"binding": 2, "visibility": vis,
             "sampler": {"type": "filtering"}},
            {"binding": 3, "visibility": vis,
             "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 4, "visibility": vis,
             "sampler": {"type": "filtering"}},
            {"binding": 5, "visibility": vis, "texture": tex3d},
            {"binding": 6, "visibility": vis,
             "sampler": {"type": "filtering"}},
            {"binding": 7, "visibility": vis, "texture": tex3d},
            {"binding": 8, "visibility": vis, "texture": tex3d},
        ])
        self._ray_pipeline_layout = d.create_pipeline_layout(
            bind_group_layouts=[self._ray_layout])
        # Group 1: the surface stages' targets. Binding 0 is the final
        # surface, read by the march; binding 1 the previous stage's, read by
        # each later stage (fs_surface_refine, fs_surface_street); binding 2
        # the veil, read by the march. They are only ever textureLoad()ed, so
        # declaring the 32-bit float formats unfilterable is right. The
        # passes that read none of them (the bake, the first surface stage)
        # are laid out without this group.
        unfilterable = {"sample_type": "unfilterable-float",
                        "view_dimension": "2d"}
        self._surface_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "uint", "view_dimension": "2d"}},
        ])
        self._march_pipeline_layout = d.create_pipeline_layout(
            bind_group_layouts=[self._ray_layout, self._surface_layout])

        self._accum_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float", "view_dimension": "2d"}},
            # The accumulator is rgba32float, which core WebGPU declares
            # unfilterable. This pass only ever textureLoad()s it; declared
            # this way, a sampled read would fail validation instead of
            # silently needing a device feature.
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float",
                         "view_dimension": "2d"}},
        ])
        self._accum_pipeline = d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self._accum_layout]),
            vertex={"module": self._accum_module, "entry_point": "vs_main"},
            fragment={"module": self._accum_module, "entry_point": "fs_main",
                      "targets": [{"format": ACCUM_FORMAT}]},
            primitive={"topology": "triangle-list"})

        # Lens water: the post pass, and the state it reads (two 1x1 texels
        # that fs_lens_state alternates between; both start dry, all zeros).
        unfilterable = {"sample_type": "unfilterable-float",
                        "view_dimension": "2d"}
        self._lens_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable}])
        lens_module = d.create_shader_module(
            code=(shader.SHADER_DIR / "lens.wgsl").read_text()
            + _LENS_POST_SHADER)
        self._lens_post_pipeline = d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self._lens_layout]),
            vertex={"module": lens_module, "entry_point": "vs_main"},
            fragment={"module": lens_module, "entry_point": "fs_main",
                      "targets": [{"format": ACCUM_FORMAT}]},
            primitive={"topology": "triangle-list"})
        self._lens_state = [
            d.create_texture(
                size=(1, 1, 1), dimension="2d", format=SURFACE_FORMAT,
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_SRC
                       | wgpu.TextureUsage.COPY_DST))
            for _ in range(2)]
        self._lens_index = 0
        self._lens_groups = None
        self._lens_out = None

    def _spec(self, *, light_cache: bool) -> dict:
        """The module's compile-time switches: this renderer's attributes
        as they stand now plus a lighting method. Every module and pipeline
        cache here keys on the whole of it, so an attribute changed after
        construction can never be served a module compiled for the old
        value."""
        return dict(periodic=self.periodic, nested=self.nested,
                    tone_map=self.tone_map,
                    light_cache=bool(light_cache))

    def _module_for(self, spec: dict):
        key = tuple(sorted(spec.items()))
        module = self._modules.get(key)
        if module is None:
            code = shader.specialize(self._shader_source, **spec)
            module = self.device.create_shader_module(code=code)
            self._modules[key] = module
        return module

    def _pipeline_for(self, entry_point: str, target_format, spec: dict,
                      layout=None):
        """`layout` defaults to the scene group alone; the march, which also
        reads the surface stages' targets, passes _march_pipeline_layout.
        `target_format` is one format, or a tuple of them for an entry point
        with several outputs (the surface stages that write the veil)."""
        formats = ((target_format,) if isinstance(target_format, str)
                   else tuple(target_format))
        key = (entry_point, formats, tuple(sorted(spec.items())))
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            module = self._module_for(spec)
            pipeline = self.device.create_render_pipeline(
                layout=layout or self._ray_pipeline_layout,
                vertex={"module": module, "entry_point": "vs_main"},
                fragment={"module": module, "entry_point": entry_point,
                          "targets": [{"format": f} for f in formats]},
                primitive={"topology": "triangle-list"})
            self._pipelines[key] = pipeline
        return pipeline

    def _ray_pipeline_for(self, view: "ViewState"):
        """The march pipeline for the lighting method `view` asks for: the
        same flag pack_uniforms writes into the block, so the block and the
        compiled method cannot disagree."""
        return self._pipeline_for("fs_main", SAMPLE_FORMAT, self._view_spec(view),
                                  layout=self._march_pipeline_layout)

    def _view_spec(self, view: "ViewState") -> dict:
        """The compile-time switches `view` asks for: its lighting method."""
        return self._spec(light_cache=view.light_cache)

    def _bake_pipeline(self):
        """The light cache's bake: one field-x plane of the cache per draw,
        into a 2D target the caller copies into the 3D cache texture.
        fs_bake_light does not read the lighting switch, so it is built on
        the default method (as the browser's is)."""
        return self._pipeline_for("fs_bake_light", LIGHT_CACHE_FORMAT,
                                  self._spec(light_cache=False))

    def _stage_pipeline_for(self, entry: str, first: bool,
                            view: "ViewState"):
        """One surface stage's pipeline with `view`'s switches: the first
        reads nothing at group 1, a later one the stage before (the march's
        layout); a VEIL_STAGES entry writes the veil too."""
        if entry in VEIL_STAGES:
            formats = (SURFACE_FORMAT, VEIL_FORMAT)
        else:
            formats = SURFACE_FORMAT
        return self._pipeline_for(
            entry, formats, self._view_spec(view),
            layout=None if first else self._march_pipeline_layout)

    def _ensure_targets(self, size):
        """This size's render targets, with the intermediate surfaces this
        surface's stage chain alternates between (surface_stage_plan). They
        are made once per size and kept, as renderer.js _ensureSurfaceTarget
        keeps the browser's."""
        entries = SURFACE_STAGES.get(self.surface, ())
        n_mid = min(max(len(entries) - 1, 0), 2)
        if self._targets is None or self._targets["size"] != size:
            has = self.surface in SURFACE_STAGES
            surface = self._make_target(size, SURFACE_FORMAT) if has else None
            veil = self._make_target(size, VEIL_FORMAT) if has else None
            self._targets = {
                "size": size, "sample": self._make_target(size, SAMPLE_FORMAT),
                "accum": [self._make_target(size, ACCUM_FORMAT),
                          self._make_target(size, ACCUM_FORMAT)],
                "surface": surface, "veil": veil,
                "plain": None, "mid": None,
                # Group 1 for a later stage reading the plain or the middle
                # surface (made with them); no stage binds what it writes.
                "after_plain": None, "after_mid": None,
                # Group 1 for the march: this size's final surface and veil.
                "surface_group": self._group1(
                    surface or self._surface_dummy, self._surface_dummy,
                    veil or self._veil_dummy),
            }
        t = self._targets
        for name, after, needed in (("plain", "after_plain", n_mid >= 1),
                                    ("mid", "after_mid", n_mid >= 2)):
            if needed and t[name] is None:
                t[name] = self._make_target(size, SURFACE_FORMAT)
            if t[name] is not None and t[after] is None:
                # A later stage's group: what it reads.
                t[after] = self._group1(self._surface_dummy, t[name],
                                        self._veil_dummy)
        return t

    def _make_target(self, size, fmt):
        """A render target."""
        wgpu = self.wgpu
        w, h = size
        usage = (wgpu.TextureUsage.RENDER_ATTACHMENT
                 | wgpu.TextureUsage.TEXTURE_BINDING
                 | wgpu.TextureUsage.COPY_SRC)
        return self.device.create_texture(
            size=(w, h, 1), dimension="2d", format=fmt, usage=usage)

    def _group1(self, final, prev, veil):
        return self.device.create_bind_group(
            layout=self._surface_layout, entries=[
                {"binding": 0, "resource": final.create_view()},
                {"binding": 1, "resource": prev.create_view()},
                {"binding": 2, "resource": veil.create_view()}])

    def _surface_stages(self, targets, view: "ViewState"):
        """The surface stages this view runs, in order, as (pipeline, group 1
        or None, the textures it writes): SURFACE_STAGES chained by
        surface_stage_plan — each later stage reads the one before at group
        1 binding 1 through the two intermediates ("plain", "mid"), the last
        writes the surface the march reads, and a VEIL_STAGES entry writes
        the veil beside it."""
        entries = SURFACE_STAGES.get(self.surface, ())
        slot = {"final": (targets["surface"], None),
                "a": (targets["plain"], targets["after_plain"]),
                "b": (targets["mid"], targets["after_mid"])}
        stages = []
        for k, (entry, (writes, reads)) in enumerate(
                zip(entries, surface_stage_plan(len(entries)))):
            out = [slot[writes][0]]
            if entry in VEIL_STAGES:
                out.append(targets["veil"])
            stages.append((self._stage_pipeline_for(entry, k == 0, view),
                           None if reads is None else slot[reads][1], out))
        return stages

    def _ensure_bind_group(self):
        if self._bind_group is not None:
            return self._bind_group
        if self._vol_tex is None:
            raise RuntimeError("No volume uploaded; call upload_volume() first.")
        self._bind_group = self.device.create_bind_group(
            layout=self._ray_layout, entries=[
                {"binding": 0, "resource": {"buffer": self._uniform_buf,
                                            "offset": 0, "size": UNIFORM_NBYTES}},
                {"binding": 1, "resource": self._vol_tex.create_view()},
                {"binding": 2, "resource": self._vol_sampler},
                {"binding": 3, "resource": self._ocean_tex.create_view()},
                {"binding": 4, "resource": self._ocean_sampler},
                {"binding": 5, "resource": self._nest_tex.create_view()},
                {"binding": 6, "resource": self._vol_wrap_sampler},
                {"binding": 7, "resource": self._light_tau_tex.create_view()},
                {"binding": 8, "resource": self._ice_frac_tex.create_view()},
            ])
        return self._bind_group

    def _check_scene(self, state: SceneState) -> None:
        """The surface, periodicity and nest are compiled in at construction.

        A SceneState that disagrees would pack the night colours into a shader
        reading them as sunlight (or the reverse), or describe one tile to a
        shader holding another, and render something plausible and wrong. So
        it is an error.
        """
        if state.surface != self.surface:
            raise RuntimeError(
                f"This renderer was built for the {self.surface} — its "
                "specialization is compiled and its tile bound — but the "
                f"scene says surface={state.surface!r}. Build a renderer for "
                "the scene you mean to render.")
        # Periodicity and the nest are compiled in the same way (the
        # PERIODIC_DOMAIN and NESTED switches); the scene's row 20.x and
        # rows 7-8 (the nest) describing something else would be packed and
        # ignored.
        if bool(state.periodic) != self.periodic or bool(state.nested) != self.nested:
            raise RuntimeError(
                f"This renderer was compiled with periodic={self.periodic}, "
                f"nested={self.nested}, but the scene says "
                f"periodic={bool(state.periodic)}, nested={bool(state.nested)}. "
                "Build a renderer for the scene you mean to render.")

    # -- the light cache -----------------------------------------------------

    @staticmethod
    def _light_cache_key_for(state: SceneState, view: ViewState) -> tuple:
        """What a baked cache is the cache of, apart from the field itself
        (an upload discards the cache): the sun, whether the domain wraps,
        and the boxes. The bake chooses its own light step, so the scene's
        is not part of it."""
        return (float(view.sun_azimuth), float(view.sun_elevation),
                bool(state.periodic),
                tuple(float(v) for v in state.bmin),
                tuple(float(v) for v in state.bmax),
                bool(state.nested)
                and (tuple(float(v) for v in state.nest_bmin),
                     tuple(float(v) for v in state.nest_bmax)))

    def bake_light_cache(self, state: SceneState, view: ViewState, *,
                         divisor: int = LIGHT_CACHE_DIVISOR
                         ) -> Tuple[int, int, int]:
        """Bake the light cache: at every cache voxel, the optical depth
        toward `view`'s sun and the optical depth of the cloud overhead.

        Every render reads the second (it is the sky visibility), and a
        render with view.light_cache=True reads the first instead of
        marching to the sun per cloud sample. render() bakes when the cache
        is not the one for its field, boxes and sun, so calling this
        directly is only needed to choose `divisor`, the linear resolution
        reduction against the volume grid (1 = one cache texel per voxel).

        The bake's sun marches step at the High tier's light step (that many
        of the level's smallest voxels), with no angular LOD, whatever
        `state.dt_light` is. The browser bakes the same way, so one cache
        serves every tier and does not depend on which render asked for it.

        The bake is one fs_bake_light draw per field-x plane, copied into the
        3D cache. Returns the cache dims (nx, ny, nz) in field axis order.
        """
        wgpu = self.wgpu
        if self._vol_tex is None:
            raise RuntimeError("No volume uploaded; call upload_volume() first.")
        self._check_scene(state)
        if not (1 <= int(divisor) <= 8):
            raise ValueError(f"divisor must be in [1, 8]; got {divisor}.")
        divisor = int(divisor)
        # Volume texture is (w=nz, h=ny, d=nx); the cache mirrors it.
        vz, vy, vx = self._vol_tex.size
        cz = max(1, -(-vz // divisor))
        cy = max(1, -(-vy // divisor))
        cx = max(1, -(-vx // divisor))

        self._light_tau_tex.destroy()
        self._light_tau_tex = self.device.create_texture(
            size=(cz, cy, cx), dimension="3d", format=LIGHT_CACHE_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._light_cache_dims = (cx, cy, cz)
        self._light_cache_key = self._light_cache_key_for(state, view)
        self._bind_group = None
        bind_group = self._ensure_bind_group()

        slice_tex = self.device.create_texture(
            size=(cz, cy, 1), dimension="2d", format=LIGHT_CACHE_FORMAT,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.COPY_SRC))
        slice_view = slice_tex.create_view()

        light_step = QUALITY_PRESETS["high"]["light_step_factor"]
        box = np.subtract(state.bmax, state.bmin)
        bake_state = dataclasses.replace(
            state, dt_light=float(np.min(box / (vx, vy, vz))) * light_step)
        if state.nested:
            nz, ny, nx = self._nest_tex.size
            nest_box = np.subtract(state.nest_bmax, state.nest_bmin)
            bake_state.dt_light_nest = (
                float(np.min(nest_box / (nx, ny, nz))) * light_step)
        base = pack_uniforms(bake_state, view)
        bake_pipeline = self._bake_pipeline()
        for ix in range(cx):
            u = base.copy()
            u[SLOT_BAKE_SLICE] = float(ix)
            self.device.queue.write_buffer(self._uniform_buf, 0, u.tobytes())
            enc = self.device.create_command_encoder()
            rp = enc.begin_render_pass(color_attachments=[{
                "view": slice_view, "load_op": "clear", "store_op": "store",
                "clear_value": (0.0, 0.0, 0.0, 1.0)}])
            rp.set_pipeline(bake_pipeline)
            rp.set_bind_group(0, bind_group)
            rp.draw(3)
            rp.end()
            enc.copy_texture_to_texture(
                {"texture": slice_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"texture": self._light_tau_tex, "mip_level": 0,
                 "origin": (0, 0, ix)},
                (cz, cy, 1))
            self.device.queue.submit([enc.finish()])
        return (cx, cy, cz)

    def _draw_banded(self, slot, pipeline, groups, clears, size,
                     submit_budget_s: float) -> None:
        """One full-frame fragment pass, drawn in horizontal scissor bands
        with one command buffer per band, waiting on the device between them
        (render() says why).

        `clears` is the pass's attachments as (texture, clear value). The
        clear is its own submit, so every band loads the target as it stands
        and keeps the others. The band height aims each submit at
        `submit_budget_s` from the measured cost of the last one, and is
        remembered per pass (`slot`) and size, so cheap frames settle to one
        band per pass.
        """
        w, h = size
        queue = self.device.queue
        views = [tex.create_view() for tex, _ in clears]
        enc = self.device.create_command_encoder()
        enc.begin_render_pass(color_attachments=[
            {"view": v, "load_op": "clear", "store_op": "store",
             "clear_value": clear}
            for v, (_, clear) in zip(views, clears)]).end()
        queue.submit([enc.finish()])
        # The first band of a new size is small, so that even a pathological
        # scene cannot run long before the host knows what it costs.
        band_rows = self._band_rows.get(
            (size, slot), max(1, min(h, INITIAL_BAND_PIXELS // max(w, 1))))
        y = 0
        while y < h:
            rows = min(band_rows, h - y)
            enc = self.device.create_command_encoder()
            rp = enc.begin_render_pass(color_attachments=[
                {"view": v, "load_op": "load", "store_op": "store"}
                for v in views])
            rp.set_pipeline(pipeline)
            for g, group in enumerate(groups):
                rp.set_bind_group(g, group)
            rp.set_scissor_rect(0, y, w, rows)
            rp.draw(3)
            rp.end()
            t0 = time.perf_counter()
            queue.submit([enc.finish()])
            # wgpu-py 0.32's on_submitted_work_done_sync is broken (its C
            # callback is declared with the wrong arity); a blocking device
            # poll is what its map_sync waits on.
            self.device._poll_wait()
            elapsed = time.perf_counter() - t0
            y += rows
            # Never more than doubling: a band of cheap sky says little about
            # the cloud rows below it.
            per_row = max(elapsed, 1e-4) / rows
            band_rows = int(max(1, min(2 * band_rows, h,
                                       submit_budget_s / per_row)))
        self._band_rows[(size, slot)] = band_rows

    # -- rendering ---------------------------------------------------------

    def render(self, state: SceneState, view: ViewState, *,
               frames: int = STILL_ACCUMULATE_FRAMES,
               lens_dt: Optional[float] = None,
               submit_budget_s: float = SUBMIT_BUDGET_S) -> np.ndarray:
        """Accumulate `frames` passes at a fixed camera and read the result.

        The camera and scene do not change across passes, so the weights are
        a plain uniform average and the result is exactly
        mean(sample_0 .. sample_{frames-1}). Pass 0 is not jittered within
        the pixel; every later pass is. The browser does the same, and a
        render with all passes jittered, or none, would differ from it.

        Each pass is drawn in horizontal bands, one command buffer per band,
        and the host waits for each band before submitting the next. This is
        what lets a slow frame finish at all: a 4K march toward a grazing sun
        is a single fragment draw that can run for tens of seconds, and
        macOS's GPU watchdog kills any command buffer that runs that long
        (IOGPUScheduler "eRestartRequest", seen on an M1; the restart also
        discards the WindowServer's work, so the whole machine freezes). The
        killed tiles come back as whatever was in memory, typically NaN.
        `submit_budget_s` is the wall-clock time each band aims for; the band
        height adapts to it from the measured cost of the last band. The
        image is the same however it is banded, because a fragment's value
        does not depend on the scissor.

        `lens_dt`, when given, advances the lens-water state by that many
        seconds and returns the image seen through the wet lens.

        Returns (h, w, 3) float64, tone-mapped to [0, 1] by the shader (or
        linear radiance from a renderer built with tone_map=False). It is
        read back from the float accumulator, so nothing is quantised on the
        way out. Raises if any pixel is non-finite: the shader cannot
        produce one, so it is always a frame the GPU did not finish.
        """
        if frames < 1:
            raise ValueError(f"frames must be >= 1; got {frames}.")
        if not submit_budget_s > 0.0:
            raise ValueError(f"submit_budget_s must be > 0; got {submit_budget_s}.")
        self._check_scene(state)
        if self._light_cache_key != self._light_cache_key_for(state, view):
            self.bake_light_cache(state, view)
        w, h = view.render_size
        targets = self._ensure_targets((w, h))
        bind_group = self._ensure_bind_group()
        sample_view = targets["sample"].create_view()
        queue = self.device.queue

        accum_index = 0
        for i in range(frames):
            pass_view = ViewState(**{**view.__dict__,
                                     "frame_index": view.frame_index + i,
                                     "subpixel": i >= 1,
                                     "jitter_scale": 1.0})
            u = pack_uniforms(state, pass_view)
            queue.write_buffer(self._uniform_buf, 0, u.tobytes())
            prev_w = 0.0 if i == 0 else i / (i + 1.0)
            samp_w = 1.0 if i == 0 else 1.0 / (i + 1.0)
            queue.write_buffer(
                self._accum_buf, 0,
                np.array([prev_w, samp_w, 0.0, 0.0], np.float32).tobytes())

            # The surface under this frame's rays, then the march over it:
            # same uniforms, same frame, same pixels, pass by pass and each
            # pass band by band.
            stages = self._surface_stages(targets, view)
            for k, (pipeline, group1, writes) in enumerate(stages):
                self._draw_banded(
                    k, pipeline,
                    [bind_group] if group1 is None else [bind_group, group1],
                    [(tex, (0.0, 0.0, 0.0, 1e30) if j == 0 else (0, 0, 0, 0))
                     for j, tex in enumerate(writes)],
                    (w, h), submit_budget_s)
            self._draw_banded(
                len(stages), self._ray_pipeline_for(view),
                [bind_group, targets["surface_group"]],
                [(targets["sample"], (0.0, 0.0, 0.0, 1.0))],
                (w, h), submit_budget_s)

            prev_tex = targets["accum"][accum_index]
            out_tex = targets["accum"][1 - accum_index]
            accum_bg = self.device.create_bind_group(
                layout=self._accum_layout, entries=[
                    {"binding": 0, "resource": {"buffer": self._accum_buf,
                                                "offset": 0, "size": 16}},
                    {"binding": 1, "resource": sample_view},
                    {"binding": 2, "resource": prev_tex.create_view()},
                ])
            enc = self.device.create_command_encoder()
            ap = enc.begin_render_pass(color_attachments=[{
                "view": out_tex.create_view(), "load_op": "clear",
                "store_op": "store", "clear_value": (0.0, 0.0, 0.0, 1.0)}])
            ap.set_pipeline(self._accum_pipeline)
            ap.set_bind_group(0, accum_bg)
            ap.draw(3)
            ap.end()
            queue.submit([enc.finish()])
            accum_index = 1 - accum_index

        final = targets["accum"][accum_index]
        if lens_dt is not None:
            final = self._lens_pass(state, view, final, lens_dt)
        img = self._read_back(final, w, h)
        bad = ~np.isfinite(img).all(axis=-1)
        if bad.any():
            rows = np.flatnonzero(bad.any(axis=1))
            raise RuntimeError(
                f"{int(bad.sum())} of {w * h} pixels came back non-finite "
                f"(rows {rows.min()}-{rows.max()}). The shader cannot produce "
                "one, so the GPU did not finish this frame — on macOS that is "
                "the GPU watchdog killing a command buffer (check `log show "
                "--predicate 'eventMessage CONTAINS \"GPURestart\"'`). Lower "
                "submit_budget_s, or the resolution.")
        return img

    # --- lens water -----------------------------------------------------------

    def reset_lens(self) -> None:
        """A dry lens (both state texels zero), for the start of a video."""
        zero = np.zeros(4, np.float32).tobytes()
        for tex in self._lens_state:
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)}, zero,
                {"offset": 0, "bytes_per_row": 16, "rows_per_image": 1},
                (1, 1, 1))

    def lens_state(self) -> np.ndarray:
        """The current lens state texel (wetness, w*, 0, 0)."""
        return self._read_back_texel(self._lens_state[self._lens_index])

    def _lens_pass(self, state: SceneState, view: ViewState, final, dt: float):
        """One step of the lens state (fs_lens_state in shader/passes.wgsl,
        dt in u.lens_dt), then the accumulated image `final` through it
        (shader/lens.wgsl). Returns the texture the post pass wrote."""
        if not (math.isfinite(dt) and dt >= 0.0):
            raise ValueError(f"lens_dt must be a finite dt >= 0; got {dt}.")
        w, h = self._targets["size"]
        if self._lens_groups is None:
            # Group 1 for the state pass: the previous state, at the binding
            # where a later surface stage reads the stage before it.
            self._lens_groups = [
                self._group1(self._surface_dummy, tex, self._veil_dummy)
                for tex in self._lens_state]
        if self._lens_out is None or self._lens_out[0] != (w, h):
            self._lens_out = ((w, h), self._make_target((w, h), ACCUM_FORMAT))
        u = pack_uniforms(state, view)
        u[SLOT_LENS_DT] = dt
        self.device.queue.write_buffer(self._uniform_buf, 0, u.tobytes())
        prev, nxt = self._lens_index, 1 - self._lens_index
        pipeline = self._pipeline_for("fs_lens_state", SURFACE_FORMAT,
                                      self._view_spec(view),
                                      layout=self._march_pipeline_layout)
        enc = self.device.create_command_encoder()
        sp = enc.begin_render_pass(color_attachments=[{
            "view": self._lens_state[nxt].create_view(), "load_op": "clear",
            "store_op": "store", "clear_value": (0.0, 0.0, 0.0, 0.0)}])
        sp.set_pipeline(pipeline)
        sp.set_bind_group(0, self._ensure_bind_group())
        sp.set_bind_group(1, self._lens_groups[prev])
        sp.draw(3)
        sp.end()
        group = self.device.create_bind_group(
            layout=self._lens_layout, entries=[
                {"binding": 0, "resource": final.create_view()},
                {"binding": 1, "resource": self._lens_state[nxt].create_view()}])
        out = self._lens_out[1]
        pp = enc.begin_render_pass(color_attachments=[{
            "view": out.create_view(), "load_op": "clear", "store_op": "store",
            "clear_value": (0.0, 0.0, 0.0, 1.0)}])
        pp.set_pipeline(self._lens_post_pipeline)
        pp.set_bind_group(0, group)
        pp.draw(3)
        pp.end()
        self.device.queue.submit([enc.finish()])
        self._lens_index = nxt
        return out

    def _read_back_texel(self, texture) -> np.ndarray:
        wgpu = self.wgpu
        buf = self.device.create_buffer(
            size=256, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": buf, "offset": 0, "bytes_per_row": 256,
             "rows_per_image": 1}, (1, 1, 1))
        self.device.queue.submit([enc.finish()])
        buf.map_sync(wgpu.MapMode.READ)
        v = np.frombuffer(bytearray(buf.read_mapped()), np.float32)[:4].copy()
        buf.unmap()
        return v

    def _read_back(self, texture, w: int, h: int) -> np.ndarray:
        """The accumulator `texture` as (h, w, 3) float64. A texture-to-
        buffer copy needs each row padded to a multiple of 256 bytes."""
        wgpu = self.wgpu
        dtype = {"rgba16float": np.float16, "rgba32float": np.float32}[ACCUM_FORMAT]
        bytes_per_texel = 4 * np.dtype(dtype).itemsize
        row = math.ceil(w * bytes_per_texel / 256) * 256
        buf = self.device.create_buffer(
            size=row * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": buf, "offset": 0, "bytes_per_row": row,
             "rows_per_image": h},
            (w, h, 1))
        self.device.queue.submit([enc.finish()])
        buf.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(bytearray(buf.read_mapped()), dtype)
        buf.unmap()
        img = raw.reshape(h, row // np.dtype(dtype).itemsize)[:, :w * 4]
        img = img.reshape(h, w, 4)
        return np.ascontiguousarray(img[:, :, :3], dtype=np.float64)

