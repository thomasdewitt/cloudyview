# Plan — soar becomes a browser tool

Status: in progress, 2026-08-04. Supersedes the desktop app once complete.

## Where it stands

Done:

- **The look crossed over, and is pinned.** Every host-side constant and
  calculation is JavaScript now (`web/soar/constants.js`, `spectral.js`,
  `camera.js`, `uniforms.js`, `field.js`), because a time-of-day slider means
  the spectral lighting cannot be baked into a template.
  `tests/test_web_uniform_parity.py` runs those modules under node and diffs
  the packed 368-byte block against `write_uniforms` across sun angles, gamma,
  fov through the pole and the sampling flags — plus the AABB on a stretched
  vertical grid, the periodic march cap, the wrapped-view notice, and the
  nest-pair rule. Perturbing one constant fails it with the row named.
- **The robustness floor** (`gpu.js`): `requiredLimits` from the adapter,
  error scopes around allocations, a lost-device handler, and refusals that
  name the axis and the number.
- **Landing page and viewer**, one page, since a `File` handle cannot survive
  a navigation.
- **The demo path end to end**: scene upload, the three-pass chain, the
  temporal accumulation state machine, flight controls, click-only menus.
- **Stills**: accumulated capture at any size, PNG download, reproduction
  metadata spliced in as a `tEXt` chunk (verified round-trip).

Not yet: ingest of arbitrary netCDF, video, bird, minimap, track recording.

## Decisions taken while building

- **Extinction and the transpose run on the CPU, in a worker** — not as a
  compute shader. `r16float` is not a storage-texture format, and
  `copyBufferToTexture`'s 256-byte row rule does not fit a `(nz+2)*2` row, so
  a compute path would need either a doubled-memory `r32float` volume or
  awkward padding. `queue.writeTexture` has no such rule. The per-voxel work
  is a multiply-add against a precomputed `rho_air(z)` — no `exp` per voxel —
  so this is cheap; revisit only if measured slow.
- **The ocean tile ships with the tool** (`web/soar/ocean/`), not with the
  demo. It is a periodic ~100 m patch of sea surface with nothing to do with
  anyone's data, it comes from a multifractal that only exists in Python, and
  a field opened from disk still needs it. ~2.8 MB, and it works offline.
- **The demo ships a zero border plus its wrap faces**, rather than the faces
  baked in, so toggling periodic is a texture write instead of a download.
- **Menus are click-only**; keys are Esc, Tab, F, F3, B, M, R, F12. The
  desktop's `menu.py` key state machine does not come across.
- **No mobile special-casing** (Thomas, 2026-08-04): the standard no-WebGPU
  message is the right answer there. The Chrome-flags advice is gated to
  desktop Linux so Android is not told to pass a command line.
- **Video is WebCodecs `VideoEncoder`**, because it takes the timestamp the
  caller gives it: a 30 s flight is 30 s of video whether a frame took 16 ms
  or six seconds to converge. Screen capture cannot do that. A frame ZIP is
  not a fallback — it is not a video (Thomas, 2026-08-04).
  Chrome has had `VideoEncoder` since 94 and Safari since 16.4; H.264 needs
  even dimensions everywhere, and Chrome matrixes RGB to YUV as **BT.601
  limited range** regardless of source, so the encoder's reported
  `decoderConfig.colorSpace` must be propagated into the container rather
  than assuming BT.709.

## The goal

One tool, in a tab. Anyone can open the demo field or their own file and
fly it; Thomas can open a multi-gigabyte turbulon run and get the same
renderer at the same framerate. Nothing installs. Nothing is packaged for
three operating systems.

**Deliverable**: a self-contained folder in this repo, `web/soar/`, that is
copied wholesale into `personal-website/thought-cloud/soar/` and synced.
Static files only — no build step at deploy time, no server. That matches
the site's URL convention (`thought-cloud/<slug>/index.html`, see the site's
`CLAUDE.md`); Cloudflare's rewrite rules serve `/thought-cloud/soar/` from
`soar/index.html`, so every asset path inside must be **relative**.

## What is already true

Worth being precise, because it is more than it looks:

- **`raymarch.wgsl` runs unmodified under WebGPU.** The web build copies it
  verbatim — md5-identical to the desktop shader — and measured **261 fps
  at 1280x720** on the RTX 5080 under Chromium (2026-07-17).
- **The look cannot drift.** `tools/export_web_demo.py` dumps the uniform
  block from a real `InteractiveRenderer`, so every spectral constant, LOD
  angle and realism gate arrives from Python. The JS host owns seven rows
  (camera, aspect, size, jitter, frame index) and nothing else.
- **soar already *is* the witness port.** `tests/test_soar_witness_parity.py`
  feeds one seeded FIF realization into both `witness.render_nested()` and
  soar and compares sky, tone map, scattering, boundary taper, and ocean.
  The browser needs soar's *capabilities*, not a new renderer.
- **The shader already supports nesting.** `soar.js` binds a 1x1x1 zero
  stand-in; the bind-group layout is the desktop one.

## Hard constraints, measured 2026-08-04

| | Chrome | Firefox |
|---|---|---|
| `maxTextureDimension3D` | **2,048** | **16,384** |
| Resident 3D texture before OOM | 7.52 GB | 12.88 GB |
| `parent` group, 2050 padded | rejected | allocates (1.79 GB) |

Chrome (Dawn) clamps to the spec floor regardless of hardware — requesting
2050, 4096, 8192 and 16384 all reject with `OperationError`, and
`--enable-webgpu-developer-features` does not move it. Firefox (wgpu)
passes the real limits through. Consequences:

1. **Never hardcode a ceiling.** Read `adapter.limits` at startup, request
   the maximum with `requiredLimits`, and when a field will not fit say
   exactly what would ("needs 2050 per axis; this browser allows 2048 —
   decimate 2x or crop to 2046"). Not a blank canvas.
2. **Demo data targets 2048** so it works everywhere.
3. **Thomas's own runs want Firefox** — full-resolution parent, no
   cropping, no bricking. Document this rather than engineering around it.
4. WebGPU on Linux needs flags in Chrome (`--enable-unsafe-webgpu
   --enable-features=Vulkan --ozone-platform=x11`; Wayland and Vulkan are
   mutually exclusive) and `dom.webgpu.enabled` in Firefox. The landing
   page must detect and explain, not fail silently.

## Gamma: one source of truth

Fixed 2026-08-04 ahead of this work. The window presented through an
`*-srgb` swapchain while `tone_map` already gamma-encoded, so the live view
double-encoded for the app's whole life (effective gamma ~3.08 against
witness's 1.4). Gamma is now a uniform (`u.periodic.w`,
`engine.DEFAULT_TONE_MAP_GAMMA = 2.66`) and the swapchain is forced to a
plain unorm format. **The browser must present to a non-sRGB canvas format
too** (`getPreferredCanvasFormat()` returns `bgra8unorm` in Chrome — check,
do not assume) and expose the same slider. Any deviation and the web build
stops matching the desktop, witness, and the parity suite at once.

## Architecture

```
web/soar/                     ← the drag-and-drop deliverable
  index.html                  ← landing: pick a demo or open a file
  app.html                    ← the viewer itself (or one page, routed)
  soar.js                     ← host: WebGPU setup, camera, uniforms
  ui.js                       ← menus, settings, dialogs
  ingest.js                   ← file open, format sniff, volume upload
  raymarch.wgsl               ← copied verbatim by the export script
  style.css
  demo/                       ← NOT committed here; fetched (see below)
```

**Demo data** is fetched from the cloudyview GitHub repo rather than
shipped with the site — `https://raw.githubusercontent.com/<user>/cloudyview/<tag>/web/demo/...`,
pinned to a tag so the deployed page never breaks when master moves. Until
the repo is pushed this stays a placeholder constant in one place
(`DEMO_BASE_URL`) with a local `demo/` fallback for development. Two
consequences to accept deliberately: raw.githubusercontent sets
`Access-Control-Allow-Origin: *` (so CORS is fine) but is not a CDN and has
rate limits — for a ~13 MB payload that is acceptable; revisit if the demo
grows.

## Feature parity inventory

Everything the desktop app does, and where it lands. **Keyboard shortcuts
only for in-flight actions** (Thomas, 2026-08-04) — menu items are
click-only, so the key state machine in `menu.py` does not get ported.

| Desktop | Browser | Notes |
|---|---|---|
| WASD / Space / Shift / C, mouse look, Tab release, scroll speed | keys | already in `soar.js` |
| F12 screenshot | key | + size presets, overlay choice, PNG download |
| ESC menu | key opens, clicks inside | |
| B bird, M minimap, F fullscreen, F3 stats | keys | in-flight |
| R record track → save → mp4 | key to start/stop | mp4 via WebCodecs; fallback = frame ZIP |
| Open file, group picker, ice prompt, units prompt | click | `showOpenFilePicker`, native dialog |
| Nest: `--nest`, add, remove | click | **shader ready, host stub today** |
| Quality tiers, render scale, smoothing, **gamma** | click | sliders |
| Time of day (presets + slider) | click | |
| Periodic toggle | click | |
| behold command | click | string + clipboard; already exists in `app.py` |
| Controls reference | click | |
| Capture settings (size, folder) | click | folder → browser download dir |

Two things do **not** cross: `behold` (Mitsuba path tracing) stays a Python
CLI, which is exactly why the browser hands you the command; and `witness`
(numba) stays as the golden reference the parity suite compares against.

## Ingest — the only genuinely hard part

**Go straight at reading netCDF-4 in the browser** (Thomas, 2026-08-04).
The intermediate converted format would work and would be quick, but it
buys a tool that only opens files you already processed — which is not the
tool. Attempt the real thing first.

`h5wasm` (libhdf5 compiled to WebAssembly) reading lazily from a
`FileSystemFileHandle`. netCDF-4 is HDF5 underneath, so the group tree,
dimensions, coordinate variables and attributes all come through the same
API the Python loader uses conceptually. Chunking means only the requested
group is read, not the 43 GB around it.

**Do this as a spike before any UI is built on it**, against
`nests_sq1km_C1003_m00.nc` — a real three-level file with compression, a
group tree, and a 1.79 GB parent. What the spike has to answer:

- Does h5wasm read chunked, deflate-compressed netCDF-4 lazily from a File
  handle, or does it want the whole file resident? (The second is fatal for
  a 43 GB run and decides everything.)
- Throughput on a ~2 GB group. 10–30 s is a load; 5 minutes is not.
- Are the coordinate/attribute conventions `io.py` relies on
  (`units`, cell centers, group discovery) reachable?

**Contingency, not a phase**: if the spike says no, generalize
`export_web_demo.py` into a `.cvvol` converter (fp16 raw + JSON header,
any group, optional decimation) and ship that as the path for large local
files while h5wasm handles the modest ones. Deciding this from a measured
spike rather than up front is the whole point of doing it first.

Either way, downstream is shared: extinction from qc and ghost padding run
as a **WebGPU compute shader** (per-voxel; no numpy needed in the browser),
then `queue.writeTexture` per z-slice so the JS heap never holds the whole
volume. The file never leaves the machine — this is not an upload.

## Phases of work

1. **Robustness floor.** `requiredLimits` from `adapter.limits`,
   `pushErrorScope` around every allocation, a `device.lost` handler, and
   the "this browser can't" page. Without this, every later failure looks
   like a blank canvas.
2. **Nest upload** (~30 lines). Unblocks the two-level runs Thomas is
   actually looking at. Highest value per line in the whole plan.
3. **The h5wasm spike.** Before anything is built on top of it — see
   above. Its answer decides phase 5's shape, so it comes early and its
   result gets written down here.
4. **Landing page + routing.** Demo picker vs. open-a-file, browser
   capability check, the copy explaining Chrome/Firefox limits.
5. **Ingest for real**, on whatever the spike proved.
6. **UI port.** Menus, settings, sun, quality, gamma, minimap, screenshot,
   behold command. The bulk of the work but the least uncertain.
7. **Capture.** High-res accumulated still + PNG download; track record and
   video.
8. **Deletion pass** (below), and `docs/architecture.md` rewritten around
   the browser as the app.

## Dead code to remove, once the browser build is complete

Only after parity is real, in one commit, so the diff is reviewable:

- `cloudyview/soar/app.py` — the imgui shell
- `cloudyview/soar/menu.py`, `hud.py`, `hud.wgsl`, `imgui_layer.py`,
  `theme.py`, `fullscreen.py`, `bird.py`+`bird.wgsl`, `track.py`,
  `jobs.py`, `__main__.py` — desktop-only, re-implemented in JS
- `cloudyview/soar/filedialog.py` — the xdg-portal dialog, written today
  and never wired; Linux-only and obsolete under this plan
- `packaging/` and the PyInstaller spec, `build/`, `dist/`
- the `interactive` extra's GUI dependencies (`glfw`, `imgui-bundle`,
  `rendercanvas`, `jeepney`) — `wgpu` stays

**Keep**: `soar/engine.py` (headless reference the parity suite runs
against), `raymarch.wgsl` (the shared artifact — still the single source of
truth for the look), `witness.py`, `behold.py`, `io.py`, and the whole test
suite. The cut is the app shell, not the Python.

## Open questions

- Track video: WebCodecs `VideoEncoder` is Chrome-solid, Firefox partial.
  Frame-sequence ZIP as the honest fallback, or drop video from v1?
- Does the demo want a second, smaller field for phones, or is "no WebGPU"
  the right answer on mobile?
- Slug: `thought-cloud/soar/` or `thought-cloud/cloudyview/`? The former
  matches what the thing is called; the latter matches the repo.
