# CloudyView architecture (2026-07 redesign)

Status: agreed direction (Thomas + Claude, 2026-07-07 session). This doc is the
spec for the library-first refactor and the interactive app.

Updated 2026-08-05: soar shipped as a desktop app first (wgpu-py + glfw +
imgui, `cloudyview/soar/`) and was rewritten for the browser. The desktop app
has been deleted; `web/soar/` is the only fly-through. Sections below are
written for the browser build. The desktop one is in git history.

## Goals

1. **Library-first.** Standard usage is Python functions: 3D cloud volume in,
   image out. CLIs remain as thin wrappers.
2. **Interactive fly-through app** (browser, WebGPU): open a .nc from the
   file picker, fly through the volume game-style, top-view minimap,
   screenshots and flight-track videos, and a copy-pasteable `behold` command
   for the current camera.
3. **Rendering tiers:** `glimpse` = 2D diagnostic; `witness` = game-like
   real-time; `behold` = offline physics engine (Mitsuba, not interactive
   even on the 5080).

## Public API (target)

```python
import cloudyview as cv

# Loading — one file with both variables, or split files (SAM LPT style
# writes one variable per file: ..._QC_*.nc, ..._QI_*.nc)
field = cv.load("cloud.nc")                          # autodetect qc + qi
field = cv.load("..._QC_0000000600.nc",
                ice="..._QI_0000000600.nc")          # split files
field = cv.load("cloud.nc", liquid_water_var="QC", ...)  # explicit overrides

field                    # CloudField dataclass
field.lwc, field.iwc     # (nx,ny,nz) float32 ndarrays, g/kg; iwc may be None
field.x, field.y, field.z  # 1D coords, meters

cam = cv.Camera(position=(0, -0.8, -0.95), azimuth=0, elevation=35, fov=100)
                         # existing conventions: met azimuth (0=N, 90=E),
                         # elevation above horizon, relative position ±1
                         # (z: -1 is the surface, not the data's floor),
                         # fov horizontal — all three renderers agree

img = cv.glimpse(field)                  # (ny,nx) two-stream visual albedo
img = cv.witness(field, camera=cam, size=(W, H))     # (H,W,3) image array
img = cv.behold(field, camera=cam, quality="high")   # (H,W,3) image array
```

Principles:
- **Library code raises exceptions; only CLI wrappers catch and `sys.exit`.**
- **No silent fallbacks, ever.** Required accelerators are required; failures
  are loud. (Longstanding project rule — see CLAUDE.md.)
- Render functions return arrays and do not write files or call matplotlib.
  Saving/plotting are separate helpers (`cv.save_image`, plotting in
  `basic_render`).
- Keep the existing CLI entry points (`glimpse`, `witness`, `behold`) working
  with identical behavior, reimplemented on top of the library API.
- Downstream consumers (`../steam-renders`, `../turbulon-analysis`) import
  cloudyview internals; check what surface they use and keep it working or
  update those repos in the same change.

## Renderer strategy

**One implementation.** `web/soar/raymarch.wgsl` is the renderer; everything
else is a host that fills its uniform block and binds its textures. There are
two hosts — JavaScript in the browser (`web/soar/renderer.js`) and Python via
wgpu (`cloudyview/soar_host.py`) — and `witness.py` is a thin wrapper around
the Python one.

It was not always this way, and the reason for the change is worth keeping.
There used to be a numba CPU kernel in `witness.py` described as "the golden
reference", with the WGSL ported function-by-function from it. Two
implementations of one look diverge: periodic domains and distance LOD landed
in the shader and never came back to the CPU, so by 2026-08 `witness` could
not render what soar renders, and the "reference" was the less capable of the
two. The kernel was deleted rather than resynchronised (Thomas's call,
2026-08-05: "it's best to have a single renderer core, and it's fine if we
lose CPU portability"). It lives in git history if ever needed.

What that costs: rendering needs a GPU. `witness --help` still does not.

What pins it: `tests/test_uniform_parity.py` diffs the Python host's 368-byte
uniform block against the browser's own `packUniforms` running under node,
byte for byte. Since the shader is shared by construction, the uniform block
and the texture upload are the only places a Python render can silently stop
matching soar — so that is what the test covers. This replaces the parity
tests deleted with the desktop app, which used the wgpu engine as their own
reference and so could not have caught a drift.

Why WGSL rather than evolving numba.cuda into the app engine: 3D-texture
hardware trilinear sampling, no per-frame host↔GPU round trip, and the shader
is the portable artifact. That portability is what let the whole engine move
to the browser: the shader crossed over verbatim and only the host had to be
rewritten in JavaScript.

Benchmark evidence (2026-07-07, RTX 5080 — the detailed writeup lived in a
scratch `temp/` folder that no longer exists; the findings that mattered are
kept here):
- witness_cuda was a stale, reduced-feature port (only 2 of the post-April
  look-tuning commits ever reached it) and was RETIRED the same day the
  WGSL spike proved out (Thomas's call; it lives in git history at tag
  `witness-cuda-final` if ever needed for archaeology).
- Both backends re-copy/re-upload the volume every frame (~230 ms/frame on
  the 1024² domain — 37–75%% of frame time). A resident texture removes this
  for free.
- On the 1 GB volume the kernel shows a memory-latency signature from manual
  8-corner trilinear loads; hardware 3D-texture sampling targets exactly
  this.
- Even best-case resident-volume CUDA is ~78 ms at 480×270 on the full
  domain: occupancy-grid empty-space skipping + progressive resolution are
  required for interactivity at 1024², whichever backend renders.
  *(2026-08-13: half right. Progressive resolution was required and shipped —
  the flight/hold ladder. Empty-space skipping was not: it has since been
  built three times and measured slower every time, on two different
  architectures. See raymarch.wgsl's note where the TODO used to be.)*

Interactive techniques (in rough order):
- Progressive rendering: reduced resolution while the camera moves, refine to
  full when still.
- Per-pixel jittered ray starts (blue-noise) + temporal accumulation — also
  the expected fix for the residual ring/banding artifact (coherent
  step-size shells around dense cores).
- ~~Coarse occupancy grid for empty-space skipping (cloud fields are
  sparse).~~ **Tried and rejected, three times** (2026-07-17 view march,
  2026-08-11 light-march majorant grid, 2026-08-13 full sparse bricks with
  page-table DDA). Correct in every case, slower in every case, on an RTX
  5080 and on Apple silicon. The march is texture-latency bound and any
  scheme that knows where the empty space is must ask something per sample to
  find out. The sparseness is real and does pay — but as *storage*, not
  traversal: see the z-crop (`web/soar/zcrop.js`), which drops the empty sky
  at load for up to 3.6x. Full post-mortem in `raymarch.wgsl`.
- fp16 density texture option for large domains.

### Nested domains

Witness composes N strictly-nested extinction grids (`render_nested`,
finest-first, absolute meters, finest level covering a point wins). Soar
implements the same model with **exactly two** levels — an outer field and
one optional finer `nest` — because that is the shape the data comes in
(a refinement run inside its parent) and because two levels need no
dynamic texture indexing, which core WebGPU does not have.

- Both volumes are resident 3D textures (bindings 1 and 5); a 1³ zero
  texture stands in when there is no nest so one bind-group layout serves
  both shader specializations.
- Placement comes from each file's own absolute coordinates. Two hard
  errors, never silent behavior:
  - a nest that does not lie inside the outer AABB — the march is clipped
    to (and wrapped into) the outer box, so an overhanging nest would be
    truncated exactly where refinement matters;
  - a nest that *fills* the outer AABB on all three axes. Finest-wins means
    it would hide the outer field entirely, which reads as "the parent
    failed to load". That is two renders of one domain, not a refinement.
  Partial coverage is reported rather than refused (`nest_coverage_fraction`,
  shown in the paused menu and printed on load) — a nest spanning the full
  horizontal domain over part of the column is legitimate, and the number
  is what explains a view with little parent left in it.
- Step size follows the active level in both the view and the light march,
  as in witness. The dt-invariant powder term is what lets levels at very
  different step scales composite without a brightness seam.
- **Periodic + nested: the whole scene is one tile.** A world point is
  wrapped into the outer domain *before* the nest containment test, so
  every domain copy carries a copy of the nest. The alternative (nest once
  in absolute space) was rejected: it makes the tiled field
  inhomogeneous and fights the distance-LOD step floor.
- The nest's ghost border stays zero even when the domain is periodic —
  that taper is how the fine field blends out into the coarse one at its
  own boundary. Only the outer level's border carries the wrap seam.
- Gradient shading pins the one-voxel fine stencil to the active level, but
  lets the wide coarse stencil re-dispatch per tap. Witness pins at every
  radius, which puts a spurious edge on the nest boundary whenever the
  coarse radius reaches past it.

Known cost: a shadow ray crossing a deep nest marches at the fine step and
can exhaust `MAX_LIGHT_STEPS` before `LIGHT_TAU_CUTOFF` saturates it. This
is the same trade witness makes. It was expected that the occupancy grid
above would fix it; that grid has since been measured and rejected three
times, so this cost stands unaddressed rather than merely pending.

## Scale targets

- `data/TWPICE_subvolume_256x256_5km.nc` — dev/test.
- `/home/thomas/Downloads/experiment/data_twpice/` — full SAM LPT TWPICE,
  1024×1024×255 per variable (~1 GB fp32). QC and QI in separate files.
  Combined extinction fits GPU memory directly; fp16 halves it. Domains
  beyond ~2048² will need bricking/LOD — out of scope for now, don't
  preclude it.

## App shell — `web/soar/`

A static WebGPU page: no server, no Python at run time. Serve `web/` and open
`soar/`. `tools/export_web_assets.py` regenerates the binary assets it ships
(the FIF ocean normal tile) and the demo field.

- Game-style pointer capture (Esc releases), WASD + Space/Shift vertical,
  scroll for speed. Camera state ↔ `cv.Camera` conventions throughout —
  meteorological angles, relative coordinates only at the edges.
- netCDF ingest in the browser (`ingest/`, h5wasm): group picker, split
  liquid/ice selection, units prompt. The questions `io.py` asks on a
  terminal get asked in the UI instead.
- Nested fields (see "Nested domains"): loaded from a second file, or from a
  second group of the same file. When one file holds several groups (STEAM
  render nests), the coordinates alone identify pairs where one lies strictly
  inside the other and is finer, and both can be loaded as one scene — picking
  one group used to mean silently losing the other. Three or more levels yield
  several pairs and the renderer holds two, so the picker lists every pair
  rather than deciding which two levels were meant. Adding a nest keeps the
  current viewpoint; opening a new *outer* file resets the camera and drops
  the nest.
- Minimap overlay: `glimpse` albedo of the loaded field, camera marker + FOV
  wedge, and — when a nest is loaded — its horizontal footprint as a subtle
  outline drawn *under* the camera overlay. A nest is easy to fly straight
  past; the outline is how you find it. Vertical extent is not shown: the
  minimap is a plan view.
- Captures share one settings block: output size and what belongs in the frame
  (bird + location map, or clouds only) — per capture rather than the live
  toggles, because the frame worth keeping and the frame worth flying with are
  rarely the same one. Stills accumulate at any size and download as PNG with
  camera, source file, sun, renderer, version and reproduction metadata in a
  `tEXt` chunk (`capture.js`, matching `cloudyview/render_metadata.py` so both
  are readable by the same tools). Flight tracks encode to video with
  WebCodecs (`video.js`, mp4 where AVC is available, else webm).
- Time-of-day panel: presets plus solar elevation and azimuth. The sun drives
  the whole spectral package (beam colour, sky field, low-sun warm wedge,
  ocean glint), so this is a look control rather than a convenience. Elevation
  is floored just above zero: a periodic domain's light march exits only
  through the domain top, so the uniform packing refuses a sun at or below the
  horizon, and clamping at the control keeps the slider usable to its end
  instead of raising at the last degree.
- **Behold is not run from the app.** A panel shows a copy-pasteable `behold`
  command for the current view — field, camera, sun, quality. The file is
  named by absolute path and by NetCDF group where the field came from one;
  with a nest loaded there are two fields on screen and behold renders one, so
  the panel asks which, and re-expresses the camera in the chosen field's box,
  since a relative position means "this far across THIS field". Path tracing
  is minutes-to-overnight and wants the GPU to itself, which does not belong
  inside a fly-through — and in the browser there is no Mitsuba at all.

## Testing

- CPU numba is truth. `witness.py` is where look-tuning lands, and
  `tests/test_behold_renders.py` pins the offline renderers against committed
  reference images.
- The browser engine is **not** currently pinned to it by a test. The desktop
  build carried two such pins — a soar/witness image comparison and a
  uniform-block diff that ran `web/soar/`'s JavaScript under node against
  `write_uniforms` — and both were deleted with the desktop app (2026-08-05),
  since each used the wgpu engine as its reference. Reinstating cover here
  means writing a reference the browser can be checked against directly:
  either frozen golden uniform blocks committed as a fixture, or headless
  WebGPU screenshots diffed against witness. Until then "the look cannot
  drift" is a hope rather than a property.
- Reference photos: Thomas's own photos get promoted to `references/photos/`
  (committed); stock-site images stay local-only (gitignored) for licensing
  reasons.
