# How cloudyview is put together

cloudyview turns a 3D field of cloud water into pictures. There are three
renderers and one app:

- **glimpse**: a 2D view from above, the two-stream albedo of each column.
  matplotlib, no GPU.
- **witness**: a volumetric ray marcher on the GPU. Fast enough to fly
  through, with lighting tuned by eye against photographs.
- **behold**: a Monte Carlo path tracer (Mitsuba 3). Physically faithful and
  very slow for clouds.
- **soar**: the browser app (`web/soar/`). It flies through a field with the
  witness renderer, and takes stills and videos.

witness and soar are one renderer. The rest of this page is about that
renderer; glimpse and behold are each one file (`glimpse.py`; `behold.py`
with `radiative_transfer.py`).

## Where things are

```
cloudyview/               the Python package
  field.py, io.py         load a NetCDF file into a CloudField
  optical_depth.py        cloud water -> extinction
  camera.py               the Camera and its angle conventions
  glimpse.py witness.py behold.py     the three renderers, each with its CLI
  track.py                `soar TRACK.json`: a recorded flight, to video
  image.py                saving images; reading a soar still's metadata
  soar/                   the renderer witness and the browser share
    shader/               the WGSL shader, in files (see below)
    ocean/, city/         the surface tiles (binary, pre-baked)
    shader.py             joins the shader files into a module
    uniforms.py           the uniform block: SceneState, ViewState, pack_uniforms
    surfaces.py           the surface tiles and their render stages
    tiers.py              the quality tiers' tables
    look.py               haze curves and the low-sun colour model
    renderer.py           SoarRenderer: the wgpu host
web/soar/                 the browser app (ES modules, no build step)
tools/                    tile and demo bakers, deploy staging, and checks
tests/
docs/
```

`web/soar/shader`, `web/soar/ocean` and `web/soar/city` are symlinks into
`cloudyview/soar/`, so there is one copy of the shader and the tiles.

## One shader, two hosts

The renderer is the WGSL under `cloudyview/soar/shader/`. A *host* compiles
it, fills its uniform block, binds its textures and draws. There are two:
JavaScript in the browser (`web/soar/renderer.js`) and Python through wgpu
(`cloudyview/soar/renderer.py`, which `witness` and `soar TRACK.json` use).
Because the shader is shared, a host can only differ from the other in what
it hands the shader, and that is what the tests pin.

### The module a host compiles

`shader/manifest.json` lists the files:

- ten shared ones: `uniforms` (what the host hands over), `math` (hashes,
  stratified sequences, phase function), `volume` (reading the field),
  `clouds` (the lighting model's constants and marches), `sky`, `ice` (the
  ice-detection view), `tonemap`, `camera`, `passes` (the small entry
  points) and `march` (`fs_main`, the cloud march);
- for each *surface* the world can stand on, the files that draw it and an
  `absent` file of stand-ins. Today the surfaces are `ocean` (`ocean.wgsl`)
  and `city` (`city/`, the night city).

A module is the shared files, one surface's files, and every other
surface's `absent` file, joined in that order. So a module holds the code of
exactly one surface. A new surface is a set of files, an `absent` file with
neutral bodies for whatever shared code calls, and an entry in the manifest.

The host puts a header of compile-time switches in front:
`PERIODIC_DOMAIN`, `NESTED`, `TONE_MAP`, `LIGHT_CACHE`. They are
constants so that the branch not taken costs nothing. `shader.py` and
`web/soar/shader.js` build the module the same way, and
`tests/test_uniform_parity.py` checks they produce the same string.

### The uniform block

One block of 21 rows of four floats, declared as the `Uniforms` struct in
`shader/uniforms.wgsl`, which says what every slot is. `pack_uniforms` in
`soar/uniforms.py` and `packUniforms` in `web/soar/uniforms.js` fill it. The
same test diffs the two packers byte for byte over about forty scenes, and
checks the struct's layout against both.

The block carries the scene and the view: camera, sun, the field's box and
step sizes, the lighting colours for this sun elevation, the haze, the
surface tile's description, the tone map. It does not carry the look's fixed
strengths; those are constants in the shader, next to the terms they scale.

## What a frame is

1. **Surface stages** (the city only): the opaque city under each ray is
   traced first, into its own target, in up to three passes (`fs_surface`,
   `fs_surface_refine`, `fs_surface_street`). The sea needs none; it is a
   plane, shaded at the end of the march.
2. **The march** (`fs_main`): one ray per pixel through the volume. At each
   cloud sample it needs the optical depth toward the sun, and combines the
   lighting terms described at the top of `clouds.wgsl`. The result is one
   noisy, tone-mapped sample of the image.
3. **Accumulation**: samples of a still view are averaged. Each frame
   jitters the ray within its pixel and the march's sampling phases, using
   sequences stratified over the accumulation (`math.wgsl`), so the average
   converges to an antialiased image.
4. **Present**: the average, with lens water and the overlays (minimap,
   the bird or paper plane), goes to the screen.

The optical depth toward the sun is the expensive part. It is either marched
per sample (`light_march_tau`) or read from the **light cache**: a 3D
texture at half the field's resolution, baked for the current sun by
`fs_bake_light` one slice at a time. Every tier but Max reads the cache for
the sun. The cache's second channel is the optical depth of the cloud
overhead, which every tier reads for how much sky a shadowed sample sees.

While the camera moves the browser renders at the tier's *flight* settings.
When it stops, it climbs a ladder of finer settings as long as each step
stays affordable (the *hold* ladder), and accumulates. A still or a video
frame is the top of that ladder, accumulated for the tier's sample count,
and `witness --quality` renders the same thing from Python.

### The tiers

| | render scale in flight | view step (voxels) | light step (voxels) | sun | haze e-folding |
|---|---|---|---|---|---|
| Max | 1.0, 8 samples a frame | 2 | 2 | marched | 70 km |
| High | 1.0 | 2 | 2 | cache | 70 km |
| Medium | 0.7 | 2.5 | 4 | cache | 25 km |
| Low | 0.3 | 3 | 8 | cache | 15 km |
| Minimal | 0.125 | 4 | 12 | cache | 8 km |

Thicker haze at the lower tiers is a performance setting as much as a look:
in a periodic domain a ray ends where the air has absorbed it, so thicker
air is a shorter march. The tables are in `web/soar/constants.js`, with the
part Python needs repeated in `soar/tiers.py`.

## Geometry is exact; lighting is tuned

This is the rule the renderer is built on. Where a cloud is, how big it
looks, what is in front of what, where its shadow falls: these come straight
from the field, the camera and the sun, with no approximation that would
make the picture say something untrue about the model output. Sampling
choices coarsen with distance at a fixed *angle*, never by cutting anything
off.

How bright and what colour each point is, on the other hand, is a stack of
approximations to multiple scattering, tuned against photographs. They are
described where they are implemented, in `clouds.wgsl` and `march.wgsl`.

## Fields in

`cv.load` (and the browser's `ingest/`) resolve which variables are liquid
and ice, which dimensions are x, y and z, and what the units are, and they
ask or refuse instead of guessing. The two implementations follow the same
rules and are tested on the same cases.

- Cloud water becomes extinction (`optical_depth.py`, `web/soar/optical.js`):
  liquid as geometric optics at a 10 um effective radius, ice by Ebert and
  Curry (1992) at 30 um.
- Empty levels above and below the cloud are cropped at load, which is where
  the sparseness of cloud fields is used.
- The volume is uploaded as a half-float 3D texture with width = z, height =
  y, depth = x, so a C-order `(nx, ny, nz)` array uploads as it lies.
- A field can be **periodic** in x and y (LES output usually is): the
  texture wraps in hardware and rays never leave sideways.
- One finer field can be **nested** inside a coarser one. The finer one wins
  where it covers, and tapers to zero at its edge.

## Checking a change

- `uv run --extra dev python -m pytest tests/` runs everything. GPU tests
  skip without a real adapter; node-driven tests run the browser's own
  modules under node.
- `tools/soar_shader_ab.py` renders a fixed set of views to raw float and
  holds them against a saved set or a second shader directory. It is how a
  change that should not move pixels shows that it did not.
- In a browser (a Chrome started with `--remote-debugging-port`):
  `tools/tint_check.mjs` compiles every module with Chrome's WGSL compiler,
  which is stricter than the one Python uses; `tools/pinned_still.mjs`
  renders stills with every input pinned and hashes them.
- `tests/reference_images/` holds eight reference views of the witness
  renderer. They guard the look; they are re-baked deliberately when the
  look is meant to change (`tests/generate_references.py`).

## Deploying soar

`docs/deploy.md`.
