# CLAUDE.md - CloudyView Project Instructions

`docs/architecture.md` says how the project is put together. Read it first.

## Python environment

This project uses **uv**. A `.venv` and `uv.lock` are present.

```bash
uv run witness cloud.nc
uv run --extra dev python -m pytest tests/ -v
```

The `dev` extra carries pytest and what the tile bakers under `tools/` need
(scaleinvariance).

## What is here

- `glimpse <file.nc>`: quick 2D visual-albedo top view (matplotlib).
- `witness <file.nc>`: volumetric ray-marched render on the GPU, through wgpu.
- `behold <file.nc> --cpu|--gpu [quality]`: path-traced render (Mitsuba 3).
- `soar <track.json>`: re-render a flight recorded in the browser app into an
  mp4 (`cloudyview/track.py`; frames piped to ffmpeg).
- Soar itself, the real-time fly-through, is a browser app under `web/soar/`:
  WebGPU, no build step, no Python at run time.

witness and soar are one renderer: the WGSL under `cloudyview/soar/shader/`
(`web/soar/shader` is a symlink to it), compiled by a Python host
(`cloudyview/soar/renderer.py`) and a JavaScript host
(`web/soar/renderer.js`). A change to the shader or to the uniform block is
a change to both.

Library API: `cv.load()` gives a `CloudField`; `cv.Camera`;
`cv.glimpse/witness/behold` return arrays. Examples in the README.

## Geometry is exact; lighting is tuned

In witness and soar, where things are and what occludes what comes straight
from the field, the camera and the sun. Do not trade that for speed or looks.
Brightness and colour are approximations tuned by eye, and are fair game.

## behold is slow

Deep path budgets are catastrophically slow on real clouds: with the old
depths (max_depth=64+, rr_depth=16+), medium and high renders of reasonable
cloud fields were on the order of months. The presets min to high therefore
truncate at max_depth<=8; only `max` keeps deep budgets. Do not quote or
estimate behold render times.

## Running soar locally

```bash
python3 tools/dev_server.py        # http://localhost:8765/soar/
```

The demo set (`web/soar/demos`) is not in this repository; it is a symlink
into the website's working tree. Without it the app still opens a file you
drop on it.

## Checking a change

```bash
uv run --extra dev python -m pytest tests/ -v
```

The whole suite takes several minutes on a laptop GPU. When reading its
result, look at every `FAILED` line, not the tail.

For a change to the shader or a host that should not move pixels:

- `tools/soar_shader_ab.py --save DIR` before, `--against DIR` after: a fixed
  set of views in raw float, compared exactly.
- `node tools/tint_check.mjs`: compiles every shader variant in Chrome, whose
  WGSL compiler is stricter than wgpu's. Needs a Chrome started with
  `--remote-debugging-port=9333`.
- `node tools/pinned_still.mjs`: stills from the browser with every input
  pinned, for a before/after hash.

`tests/reference_images/` holds golden images for witness/soar (eight views)
and behold. They must be baked on real GPU hardware; the generator refuses a
software rasterizer. Re-bake one set with `tests/generate_references.py
--target soar|behold`, and only when the look is meant to have changed.

## Deploying

`tools/stage_deploy.py --clean` writes `dist/soar/`; `docs/deploy.md` has the
rest. The site itself is deployed by Thomas from the website repository.

## Data

`data/` holds test fields (`TWPICE_subvolume_*.nc`, real cloud from the
TWP-ICE campaign) and the ice scattering models behold uses. The demo
sources, when present, are in `data/demos/`.
