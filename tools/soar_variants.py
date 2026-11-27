"""Shared machinery for tools that render or time patched variants of the
soar shader: tools/soar_void_census.py and tools/soar_ablate.py.

Nothing here changes the shader in the tree. A variant is the module text
(cloudyview.soar.shader.assemble) with text patches applied, each anchored on
a snippet that must occur exactly once, so a patch whose anchor has moved
fails loudly instead of compiling into something else.

What is here:
- patch(): one anchored text substitution.
- VariantRenderer: SoarRenderer holding several shader sources on one device
  and one uploaded field, switched by name, with optional float32 output (for
  counts), an optional texture at the march's group 1 binding 0 (unused over
  the ocean, so free to carry per-pixel data), a zero field of the same shape
  as the real one, and GPU timestamps of every banded fragment pass.
- The look a tier flies with in the browser (haze, the two LOD angles, the
  light cache), and the views: the five soar_bench day views and five
  flight-like views placed from the field's own cloud levels.
- time_variants(): wall-clock and GPU time of a list of variants on one view,
  interleaved so that drift in the GPU's state falls on all of them alike.
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "tests", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cloudyview.soar import renderer as soar_renderer          # noqa: E402
from cloudyview.soar import shader                            # noqa: E402
from cloudyview.soar.renderer import SoarRenderer             # noqa: E402
from cloudyview.soar.surfaces import (SURFACE_STAGES,          # noqa: E402
                                      surface_cell_m,
                                      surface_clock_period_s,
                                      surface_period_m)
from cloudyview.soar.uniforms import (APP_LIGHT_MARCH_LOD_DEGREES,  # noqa: E402
                                      APP_VIEW_STEP_LOD_DEGREES,
                                      DEFAULT_OCEAN_REFLECTANCE,
                                      SceneState, ViewState,
                                      camera_world_origin)
from soar_bench import (DAY_VIEWS, TIER_ORDER, flight_presets,  # noqa: E402
                        level_from_demo)

__all__ = ["REPO", "patch", "VariantRenderer", "make_device", "load_demo",
           "tier_scene", "tier_look", "demo_views", "time_variants",
           "flight_presets", "TIER_ORDER", "shader"]


def patch(src: str, anchor: str, replacement: str, *, what: str) -> str:
    """`src` with the one occurrence of `anchor` replaced. Raises when the
    anchor occurs zero or several times: the shader has moved under the
    patch, and the variant would not be the one asked for."""
    n = src.count(anchor)
    if n != 1:
        raise RuntimeError(f"patch '{what}': anchor occurs {n} times, expected "
                           f"exactly once. Anchor:\n{anchor}")
    return src.replace(anchor, replacement)


# -- the device ---------------------------------------------------------------

def make_device(*, timestamps: bool, float32_filterable: bool):
    """A high-performance adapter's device with the features asked for.
    `timestamps` is a preference (the tools fall back to wall-clock time
    without it); `float32_filterable` is a requirement (the census reads
    float32 counts through the accumulation pass)."""
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    feats = set(adapter.features)
    want = []
    if timestamps and "timestamp-query" in feats:
        want.append("timestamp-query")
    if float32_filterable:
        if "float32-filterable" not in feats:
            raise SystemExit("this adapter has no float32-filterable feature, "
                             "which the census's float32 counts need")
        want.append("float32-filterable")
    device = adapter.request_device_sync(required_features=want)
    return adapter, device


# -- the renderer ---------------------------------------------------------------

class VariantRenderer(SoarRenderer):
    """A SoarRenderer with several named shader sources.

    The parent caches modules and pipelines by compile-time switches only;
    here the variant's name is part of every key, so switching variants
    recompiles nothing that was compiled before.
    """

    MAX_TIMESTAMPS = 4096

    def __init__(self, sources: dict, *, device, periodic: bool,
                 surface: str = "ocean"):
        if "stock" not in sources:
            raise ValueError("sources must hold the unpatched module as 'stock'")
        self._sources = dict(sources)
        self._variant = "stock"
        self._f32_variants = set()
        self._group1_tex = None
        self._group1_group = None
        self._real_tex = None
        self._zero_tex = None
        self._ts_on = False
        super().__init__(periodic=periodic, nested=False, surface=surface,
                         device=device, shader_source=self._sources["stock"])
        self.timestamps = "timestamp-query" in set(device.features)
        if self.timestamps:
            wgpu = self.wgpu
            self._qs = device.create_query_set(type="timestamp",
                                               count=self.MAX_TIMESTAMPS)
            self._qbuf = device.create_buffer(
                size=8 * self.MAX_TIMESTAMPS,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
            self._qread = device.create_buffer(
                size=8 * self.MAX_TIMESTAMPS,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        self.march_slot = len(SURFACE_STAGES.get(self.surface, ()))

    # variants
    def add_variant(self, name: str, source: str, *, f32_output: bool = False):
        if f32_output and "float32-filterable" not in set(self.device.features):
            raise RuntimeError("float32 output needs the float32-filterable "
                               "feature on the device")
        self._sources[name] = source
        if f32_output:
            self._f32_variants.add(name)

    def set_variant(self, name: str, *, group1_texture=None):
        """Select the source `name`. `group1_texture`, an rgba32float texture
        of the render size, replaces the march's group 1 binding 0 (the
        traced surface, which no ocean module reads)."""
        if name not in self._sources:
            raise KeyError(name)
        if group1_texture is not None and self.surface in SURFACE_STAGES:
            raise RuntimeError("group 1 binding 0 is the traced surface on "
                               f"{self.surface}; it cannot carry other data")
        self._variant = name
        if group1_texture is not self._group1_tex:
            self._group1_tex = group1_texture
            self._group1_group = None

    @property
    def variant(self):
        return self._variant

    def _module_for(self, spec: dict):
        key = (self._variant, tuple(sorted(spec.items())))
        module = self._modules.get(key)
        if module is None:
            code = shader.specialize(self._sources[self._variant], **spec)
            module = self.device.create_shader_module(code=code)
            self._modules[key] = module
        return module

    def _pipeline_for(self, entry_point: str, target_format, spec: dict,
                      layout=None):
        formats = ((target_format,) if isinstance(target_format, str)
                   else tuple(target_format))
        key = (self._variant, entry_point, formats, tuple(sorted(spec.items())))
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

    def _sample_format(self):
        return ("rgba32float" if self._variant in self._f32_variants
                else soar_renderer.SAMPLE_FORMAT)

    def _ray_pipeline_for(self, view):
        return self._pipeline_for("fs_main", self._sample_format(),
                                  self._view_spec(view),
                                  layout=self._march_pipeline_layout)

    def _ensure_targets(self, size):
        t = super()._ensure_targets(size)
        if "sample_stock" not in t:
            t["sample_stock"] = t["sample"]
            t["sample_f32"] = None
            t["group1_stock"] = t["surface_group"]
        if self._variant in self._f32_variants:
            if t["sample_f32"] is None:
                t["sample_f32"] = self._make_target(size, "rgba32float")
            t["sample"] = t["sample_f32"]
        else:
            t["sample"] = t["sample_stock"]
        if self._group1_tex is None:
            t["surface_group"] = t["group1_stock"]
        else:
            if self._group1_group is None:
                self._group1_group = self._group1(
                    self._group1_tex, self._surface_dummy, self._veil_dummy)
            t["surface_group"] = self._group1_group
        return t

    def data_texture(self, arr: np.ndarray):
        """An rgba32float texture holding (h, w, 4) `arr`, for set_variant's
        group1_texture."""
        wgpu = self.wgpu
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        h, w, c = arr.shape
        if c != 4:
            raise ValueError("data texture must be (h, w, 4)")
        tex = self.device.create_texture(
            size=(w, h, 1), dimension="2d", format="rgba32float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)}, arr.tobytes(),
            {"offset": 0, "bytes_per_row": 16 * w, "rows_per_image": h},
            (w, h, 1))
        return tex

    def read_sample(self) -> np.ndarray:
        """The last pass's march output, all four channels, (h, w, 4) float32.
        After render(frames=1) it is that one frame exactly."""
        w, h = self._targets["size"]
        tex = self._targets["sample"]
        fmt = self._sample_format()
        dtype = {"rgba16float": np.float16, "rgba32float": np.float32}[fmt]
        bpt = 4 * np.dtype(dtype).itemsize
        row = math.ceil(w * bpt / 256) * 256
        wgpu = self.wgpu
        buf = self.device.create_buffer(
            size=row * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": buf, "offset": 0, "bytes_per_row": row,
             "rows_per_image": h}, (w, h, 1))
        self.device.queue.submit([enc.finish()])
        buf.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(bytearray(buf.read_mapped()), dtype)
        buf.unmap()
        buf.destroy()
        img = raw.reshape(h, row // np.dtype(dtype).itemsize)[:, :w * 4]
        return np.ascontiguousarray(img.reshape(h, w, 4), dtype=np.float32)

    # the zero field
    def upload_volume(self, sigma: np.ndarray) -> None:
        if self._zero_tex is not None:
            self._vol_tex = self._real_tex
            self._zero_tex.destroy()
            self._zero_tex = None
        super().upload_volume(sigma)
        self._real_tex = self._vol_tex
        self._zero_shape = sigma.shape

    def use_zero_field(self, on: bool) -> None:
        """Swap the bound field for one of the same shape holding zero
        everywhere (WebGPU zero-initializes a texture that was never
        written), or back. The light cache is left as the real field's: with
        no cloud sample anywhere nothing reads it."""
        if on and self._zero_tex is None:
            nx, ny, nz = self._zero_shape
            self._zero_tex = self.device.create_texture(
                size=(nz, ny, nx), dimension="3d", format="r16float",
                usage=(self.wgpu.TextureUsage.TEXTURE_BINDING
                       | self.wgpu.TextureUsage.COPY_DST))
        target = self._zero_tex if on else self._real_tex
        if self._vol_tex is not target:
            self._vol_tex = target
            self._bind_group = None

    def bake_light_cache(self, state, view, **kw):
        if self._zero_tex is not None and self._vol_tex is self._zero_tex:
            raise RuntimeError("refusing to bake the light cache of the zero "
                               "field: render the real field first")
        return super().bake_light_cache(state, view, **kw)

    # GPU timestamps
    def start_gpu_timing(self):
        if not self.timestamps:
            return
        self._ts_on = True
        self._ts_next = 0
        self._ts_slots = []

    def stop_gpu_timing(self) -> Optional[dict]:
        """GPU milliseconds of the banded passes since start_gpu_timing, by
        slot (the march is `march_slot`). Timestamps are read as
        nanoseconds, which is what WebGPU specifies."""
        if not self.timestamps:
            return None
        self._ts_on = False
        n = self._ts_next
        if n == 0:
            return {}
        enc = self.device.create_command_encoder()
        enc.resolve_query_set(self._qs, 0, n, self._qbuf, 0)
        enc.copy_buffer_to_buffer(self._qbuf, 0, self._qread, 0, 8 * n)
        self.device.queue.submit([enc.finish()])
        self._qread.map_sync(self.wgpu.MapMode.READ, 0, 8 * n)
        ts = np.frombuffer(bytearray(self._qread.read_mapped(0, 8 * n)),
                           np.uint64).astype(np.int64)
        self._qread.unmap()
        out = {}
        for k, slot in enumerate(self._ts_slots):
            out[slot] = out.get(slot, 0.0) + (ts[2 * k + 1] - ts[2 * k]) * 1e-6
        return out

    def _draw_banded(self, slot, pipeline, groups, clears, size,
                     submit_budget_s: float) -> None:
        """The parent's banded pass, with a timestamp pair on each band while
        GPU timing is on. The banding logic is the parent's, line for line."""
        if not self._ts_on:
            return super()._draw_banded(slot, pipeline, groups, clears, size,
                                        submit_budget_s)
        w, h = size
        queue = self.device.queue
        views = [tex.create_view() for tex, _ in clears]
        enc = self.device.create_command_encoder()
        enc.begin_render_pass(color_attachments=[
            {"view": v, "load_op": "clear", "store_op": "store",
             "clear_value": clear}
            for v, (_, clear) in zip(views, clears)]).end()
        queue.submit([enc.finish()])
        band_rows = self._band_rows.get(
            (size, slot), max(1, min(h, soar_renderer.INITIAL_BAND_PIXELS
                                     // max(w, 1))))
        y = 0
        while y < h:
            rows = min(band_rows, h - y)
            if self._ts_next + 2 > self.MAX_TIMESTAMPS:
                raise RuntimeError("out of timestamp queries")
            k = self._ts_next
            self._ts_next += 2
            self._ts_slots.append(slot)
            enc = self.device.create_command_encoder()
            rp = enc.begin_render_pass(
                color_attachments=[
                    {"view": v, "load_op": "load", "store_op": "store"}
                    for v in views],
                timestamp_writes={"query_set": self._qs,
                                  "beginning_of_pass_write_index": k,
                                  "end_of_pass_write_index": k + 1})
            rp.set_pipeline(pipeline)
            for g, group in enumerate(groups):
                rp.set_bind_group(g, group)
            rp.set_scissor_rect(0, y, w, rows)
            rp.draw(3)
            rp.end()
            t0 = time.perf_counter()
            queue.submit([enc.finish()])
            self.device._poll_wait()
            elapsed = time.perf_counter() - t0
            y += rows
            per_row = max(elapsed, 1e-4) / rows
            band_rows = int(max(1, min(2 * band_rows, h,
                                       submit_budget_s / per_row)))
        self._band_rows[(size, slot)] = band_rows


# -- scenes, looks and views -------------------------------------------------------

DEMOS_DIR = REPO / "web" / "soar" / "demos"


def load_demo(demo: str, root=None):
    """(level, meta) of a baked demo: a directory, or a name under `root`
    (default web/soar/demos, which in a checkout is a symlink into the
    website's tree and may be absent, as in a second worktree)."""
    root = Path(root) if root else DEMOS_DIR
    d = Path(demo)
    if not (d / "meta.json").exists():
        d = root / demo
    if not (d / "meta.json").exists():
        raise SystemExit(f"no baked demo at {demo} (looked for meta.json there "
                         f"and under {root}; pass --demo-dir)")
    meta = json.loads((d / "meta.json").read_text())
    return level_from_demo(d), meta


def tier_look(presets: dict, tier: str, *, city: bool = False) -> dict:
    """The ViewState keywords tier `tier` flies with in the browser
    (web/soar/viewer/loop.js _viewKwargs; constants.js tierLodStrength):
    the tier's haze, uniform haze, and the two LOD angles scaled by the
    tier's LOD strength."""
    p = presets[tier]
    lod = p["cityLodStrength"] if city else p["lodStrength"]
    return dict(light_cache=bool(p["lightCache"]),
                ahead_march=bool(p["aheadMarch"]),
                fine_gradient=bool(p["fineGradient"]), haze=float(p["haze"]),
                haze_height_dependent=False,
                light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES * lod,
                view_step_lod_degrees=APP_VIEW_STEP_LOD_DEGREES * lod)


def tier_scene(renderer: SoarRenderer, level, presets: dict, tier: str,
               periodic: bool) -> SceneState:
    """The SceneState of `tier`'s flight sampling over `level`."""
    voxel = float(min(level.dx))
    p = presets[tier]
    meta = renderer.surface_meta
    surface = renderer.surface
    return SceneState(
        bmin=[float(v) for v in level.bmin], bmax=[float(v) for v in level.bmax],
        dt_view=voxel * p["stepFactor"], dt_light=voxel * p["lightStepFactor"],
        periodic=bool(periodic), ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE,
        surface=surface, ocean_fif_dx=surface_cell_m(surface, meta),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1,
        surface_period_m=surface_period_m(surface, meta),
        clock_period_s=surface_clock_period_s(surface, meta))


def render_size(presets: dict, tier: str, out_w: int, out_h: int):
    s = presets[tier]["renderScale"]
    return (max(1, round(out_w * s)), max(1, round(out_h * s)))


def cloud_levels(level, *, thick: float = 1e-3, faint: float = 1e-5,
                 cover: float = 0.01) -> dict:
    """Heights (m) read off the field's column cover: `base` and `top` are
    the lowest and highest levels where at least `cover` of the columns
    hold sigma > `thick` (1/m); `top_any` the highest where `cover` hold
    anything above `faint`. Cover is estimated on every other column."""
    s = level.sigma[::2, ::2, :]
    nz = s.shape[2]
    dz = (level.bmax[2] - level.bmin[2]) / nz
    zc = level.bmin[2] + (np.arange(nz) + 0.5) * dz
    cf_thick = np.array([float((s[:, :, k] > thick).mean()) for k in range(nz)])
    cf_faint = np.array([float((s[:, :, k] > faint).mean()) for k in range(nz)])
    on = np.flatnonzero(cf_thick >= cover)
    on_any = np.flatnonzero(cf_faint >= cover)
    if on.size == 0:
        raise SystemExit(f"{level.name}: no level reaches {cover} cover of "
                         f"sigma > {thick}; cannot place the flight views")
    return dict(base=float(zc[on[0]]), top=float(zc[on[-1]]),
                top_any=float(zc[on_any[-1]]), dz=float(dz))


def clear_xy(level, z: float, periodic: bool, *, grid: int = 48,
             radius_vox: int = 4):
    """A camera xy at height z in the clearest air available: on a grid of
    candidates, the one whose surroundings (radius_vox voxels laterally,
    one level up and down) have the smallest peak sigma, nearest the
    domain's centre among ties. Returns (x, y, peak sigma there)."""
    from scipy.ndimage import maximum_filter
    nx, ny, nz = level.sigma.shape
    dz = (level.bmax[2] - level.bmin[2]) / nz
    cx = 0.5 * (level.bmin[0] + level.bmax[0])
    cy = 0.5 * (level.bmin[1] + level.bmax[1])
    k = int(math.floor((z - level.bmin[2]) / dz))
    if k < 0 or k >= nz:
        return cx, cy, 0.0
    slab = np.asarray(level.sigma[:, :, max(k - 1, 0):k + 2],
                      dtype=np.float32).max(axis=2)
    local = maximum_filter(slab, size=2 * radius_vox + 1,
                           mode="wrap" if periodic else "nearest")
    lo, hi = (0.0, 1.0) if periodic else (0.15, 0.85)
    fr = lo + (np.arange(grid) + 0.5) / grid * (hi - lo)
    best = None
    for fx in fr:
        for fy in fr:
            i = min(int(fx * nx), nx - 1)
            j = min(int(fy * ny), ny - 1)
            v = float(local[i, j])
            d = (fx - 0.5) ** 2 + (fy - 0.5) ** 2
            if best is None or (v, d) < (best[0], best[1]):
                best = (v, d, fx, fy)
    v, _, fx, fy = best
    x = level.bmin[0] + fx * (level.bmax[0] - level.bmin[0])
    y = level.bmin[1] + fy * (level.bmax[1] - level.bmin[1])
    return float(x), float(y), v


def demo_views(level, meta: dict, periodic: bool) -> tuple:
    """(views, placement): name -> ViewState keywords (camera in world
    metres, the demo's own sun), and the heights the flight views were
    placed at.

    bench_*: soar_bench's five day views.
    fly_cruise:     at base + 0.35 (top - base), the clearest air there,
                    looking level across the sun (sun azimuth + 90).
    fly_cruise_back: the same point, sun behind (azimuth + 180), 5 deg down.
    fly_under_up:   at half the base height, centre of the box, 25 deg up,
                    sun azimuth + 45.
    fly_above_down: top_any + 0.1 (top_any - base), centre, 20 deg down,
                    sun azimuth + 135.
    fly_demo_still: the demo's own opening camera (meta.json "still").
    """
    sun = dict(sun_azimuth=float(meta["sun"]["azimuth"]),
               sun_elevation=float(meta["sun"]["elevation"]))
    sa = sun["sun_azimuth"]
    views = {}
    for name, v in DAY_VIEWS.items():
        pos = camera_world_origin(v["camera_position"], level.bmin, level.bmax)
        views[f"bench_{name}"] = dict(camera_position=[float(p) for p in pos],
                                      azimuth=v["azimuth"],
                                      elevation=v["elevation"], fov=v["fov"],
                                      **sun)
    lv = cloud_levels(level)
    z_cruise = lv["base"] + 0.35 * (lv["top"] - lv["base"])
    z_under = 0.5 * lv["base"]
    z_above = lv["top_any"] + 0.1 * (lv["top_any"] - lv["base"])
    xc, yc, peak = clear_xy(level, z_cruise, periodic)
    cx = 0.5 * (level.bmin[0] + level.bmax[0])
    cy = 0.5 * (level.bmin[1] + level.bmax[1])
    views["fly_cruise"] = dict(camera_position=[xc, yc, z_cruise],
                               azimuth=(sa + 90) % 360, elevation=0.0,
                               fov=75.0, **sun)
    views["fly_cruise_back"] = dict(camera_position=[xc, yc, z_cruise],
                                    azimuth=(sa + 180) % 360, elevation=-5.0,
                                    fov=75.0, **sun)
    views["fly_under_up"] = dict(camera_position=[cx, cy, z_under],
                                 azimuth=(sa + 45) % 360, elevation=25.0,
                                 fov=80.0, **sun)
    views["fly_above_down"] = dict(camera_position=[cx, cy, z_above],
                                   azimuth=(sa + 135) % 360, elevation=-20.0,
                                   fov=75.0, **sun)
    st = meta["still"]["camera"]
    pos = camera_world_origin(st["position"], level.bmin, level.bmax)
    views["fly_demo_still"] = dict(camera_position=[float(p) for p in pos],
                                   azimuth=float(st["azimuth"]),
                                   elevation=float(st["elevation"]),
                                   fov=float(st["fov"]), **sun)
    placement = dict(**lv, z_cruise=z_cruise, z_under=z_under,
                     z_above=z_above, cruise_xy=[xc, yc],
                     cruise_peak_sigma=peak)
    return views, placement


def make_view(view_kw: dict, look: dict, out_size, size, **extra) -> ViewState:
    return ViewState(output_size=tuple(out_size), render_size=tuple(size),
                     jitter=True, **look, **view_kw, **extra)


# -- timing -------------------------------------------------------------------

@dataclass
class Arm:
    """One timed configuration: a variant, the field it runs on, and the
    data texture at group 1 binding 0 if it reads one."""
    label: str
    variant: str
    zero_field: bool = False
    group1_texture: object = None


def _select(renderer: VariantRenderer, arm: Arm):
    renderer.set_variant(arm.variant, group1_texture=arm.group1_texture)
    renderer.use_zero_field(arm.zero_field)


def time_variants(renderer: VariantRenderer, state, view, arms, *,
                  frames: int, repeats: int) -> dict:
    """label -> {"wall_ms": [...], "gpu_ms": [...]} per frame, one entry per
    repeat. Each arm is warmed (compiled, band height settled) first; then
    every repeat times every arm once, in an order rotated by one each
    repeat, so that a slow spell of the GPU falls on all arms alike.

    wall_ms is soar_bench's number: render(frames) / frames, which includes
    one read-back of the accumulator. gpu_ms is the march pass alone, from
    timestamps, or None without the timestamp-query feature; gpu_all_ms is
    every timestamped (banded) pass, the surface stages with the march."""
    for arm in arms:
        _select(renderer, arm)
        renderer.render(state, view, frames=2)
    out = {a.label: {"wall_ms": [], "gpu_ms": [], "gpu_all_ms": []} for a in arms}
    for rep in range(repeats):
        order = arms[rep % len(arms):] + arms[:rep % len(arms)]
        for arm in order:
            _select(renderer, arm)
            renderer.start_gpu_timing()
            t0 = time.perf_counter()
            renderer.render(state, view, frames=frames)
            wall = 1e3 * (time.perf_counter() - t0) / frames
            gpu = renderer.stop_gpu_timing()
            out[arm.label]["wall_ms"].append(wall)
            out[arm.label]["gpu_ms"].append(
                None if gpu is None
                else gpu.get(renderer.march_slot, 0.0) / frames)
            out[arm.label]["gpu_all_ms"].append(
                None if gpu is None else sum(gpu.values()) / frames)
    _select(renderer, arms[0])
    renderer.use_zero_field(False)
    return out


def summarize(samples: list) -> dict:
    """Fastest, and spread = (slowest - fastest) / fastest."""
    xs = [x for x in samples if x is not None]
    if not xs:
        return {"min": None, "spread": None}
    lo, hi = min(xs), max(xs)
    return {"min": lo, "spread": (hi - lo) / lo if lo > 0 else None}
