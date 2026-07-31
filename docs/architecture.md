# CloudyView architecture (2026-07 redesign)

Status: agreed direction (Thomas + Claude, 2026-07-07 session). This doc is the
spec for the library-first refactor and the interactive app.

## Goals

1. **Library-first.** Standard usage is Python functions: 3D cloud volume in,
   image out. CLIs remain as thin wrappers.
2. **Interactive fly-through app** (desktop, this machine, RTX 5080): open a
   .nc via file dialog, fly through the volume game-style, top-view minimap,
   screenshots, launch a `behold` render from the current camera.
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

Two implementations of the *same* witness look:

1. **numba CPU** (`witness.py`) — the golden reference. All look-tuning lands
   here first. ~1300 lines, painstakingly tuned against real cloud photos.
2. **WGSL (wgpu-py)** — the interactive engine, ported function-by-function
   from the numba kernel and verified against numba golden images (extend the
   existing CPU/CUDA equivalence-test pattern; tolerance-based, since GPU
   float math differs slightly). The numba implementation is the mirror the
   shader is developed against — port + verify, never re-tune from scratch.

Why WGSL/wgpu rather than evolving numba.cuda into the app engine: 3D-texture
hardware trilinear sampling, no per-frame Python↔GPU round trip, and the
shader is the portable artifact for the eventual browser/WebGPU direction.
Python stays the host language (windowing, IO, xarray stack); the
performance-critical inner loop leaves Python either way.

Benchmark evidence (2026-07-07, RTX 5080 — details in
temp/benchmarks-2026-07-07/RESULTS.md):
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

Interactive techniques (in rough order):
- Progressive rendering: reduced resolution while the camera moves, refine to
  full when still.
- Per-pixel jittered ray starts (blue-noise) + temporal accumulation — also
  the expected fix for the residual ring/banding artifact (coherent
  step-size shells around dense cores).
- Coarse occupancy grid for empty-space skipping (cloud fields are sparse).
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
is the same trade witness makes; the occupancy grid above is the fix.

## Scale targets

- `data/TWPICE_subvolume_256x256_5km.nc` — dev/test.
- `/home/thomas/Downloads/experiment/data_twpice/` — full SAM LPT TWPICE,
  1024×1024×255 per variable (~1 GB fp32). QC and QI in separate files.
  Combined extinction fits GPU memory directly; fp16 halves it. Domains
  beyond ~2048² will need bricking/LOD — out of scope for now, don't
  preclude it.

## App shell (v0) — `cloudyview/soar/`, CLI `soar`

- wgpu-py surface in a simple window (glfw via wgpu-py's gui module).
- Game-style pointer capture (glfw CURSOR_DISABLED; Tab releases), WASD +
  Space/LShift vertical.
- File-open dialog for .nc selection (split liquid/ice selection supported).
  Opens at `$HOME` and then remembers the last directory of the session.
- Nested fields (see "Nested domains") three ways:
  - `--nest FILE` at launch;
  - **N** in the ESC menu, which reuses the whole open-file chain — same
    browser, group picker, ice prompt, units prompt — with one flag deciding
    whether the loaded field replaces the scene or becomes its nest. N flips
    to "Remove nest" once one is loaded;
  - **"Use both, nested"** in the group picker. When one file holds several
    groups (STEAM render nests), `io.find_nestable_group_pair` probes their
    *coordinates only* for a pair where one lies strictly inside the other
    and is finer, and offers to load both as one scene. This is the common
    case, and picking one group used to mean silently losing the other.
  Adding a nest keeps the current viewpoint (you are already flying in that
  scene); opening a new *outer* file resets the camera and drops the nest.
- WASD + mouse-look camera, scroll for speed; camera state ↔ `cv.Camera`.
- Minimap overlay: `cv.glimpse` albedo of the loaded field, camera marker +
  FOV wedge (reuse glimpse overlay math).
- Screenshot key (F12): prompts for what belongs in the frame (bird +
  location map, or clouds only), then writes an offscreen PNG at the current
  window size with camera, source-file, sun, renderer, version, timestamp,
  and reproduction metadata embedded in PNG text chunks. The choice is per
  shot rather than the live B/M toggles — the frame worth keeping and the
  frame worth flying with are rarely the same one.
- ESC menu as control center: open a new `.nc` (with split ice-file prompt),
  render the current `app.camera()` in `behold`, toggle fullscreen, resume, or
  quit. Behold runs in the foreground because the Mitsuba GPU backend needs the
  full device; the title reports progress/ETA and warns that it cannot be
  canceled once started.
- Later garnish: a subject (bird / paper airplane) in front of the camera.

## Testing

- Keep and extend the equivalence-test pattern: CPU numba is truth; CUDA and
  WGSL match within tolerance on benchmark scenes.
- Reference photos: Thomas's own photos get promoted to `references/photos/`
  (committed); stock-site images stay local-only (gitignored) for licensing
  reasons.
