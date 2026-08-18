# CLAUDE.md - CloudyView Project Instructions

## Python Environment

This project uses **uv** for dependency management. A `.venv` and `uv.lock` are already present.

When running Python commands via Bash, use `uv run`:

```bash
uv run python your_command_here
uv run witness cloud.nc
```

For commands that need optional dependencies (tests):

```bash
uv run --extra dev python -m pytest tests/ -v
```

## Project Overview

CloudyView is a 3D cloud field visualization toolkit with radiative transfer capabilities. It uses Mitsuba 3 for Monte Carlo path tracing with physically-based sky models and Mie scattering phase functions.

## Key Commands

- `behold <file.nc> --cpu|--gpu [quality]` - Photorealistic path-traced render
  (Mitsuba 3). Deep path budgets are catastrophically slow on real clouds:
  with the old depths (max_depth=64+, rr_depth=16+), medium and high renders
  of reasonable cloud fields were on the order of MONTHS (Thomas, 2026-08-18:
  "I am seeing months for medium and high, order of magnitude"). The presets
  min–high therefore truncate at max_depth≤8; only 'max' keeps deep budgets.
  Never quote or estimate behold render times; if depth settings come up,
  assume intractability until Thomas's own timings say otherwise.
- `glimpse <file.nc>` - Quick 2D visual-albedo top view
- `witness <file.nc>` - Volumetric ray-marched render. Drives the same
  `cloudyview/soar/raymarch.wgsl` the browser does (web/soar/raymarch.wgsl is
  a symlink to it), via wgpu, so there is one renderer core and one
  definition of the look. Needs a GPU.

Soar, the real-time fly-through, is a browser app under `web/soar/` — WebGPU, no
Python at run time. Serve the `web/` directory and open `soar/`:

```bash
python3 -m http.server 8765 --directory web
```

`tools/export_web_assets.py` regenerates its binary assets (the FIF ocean tile
and the demo field).

Library API (preferred): `cv.load()` → `CloudField`, `cv.Camera`, `cv.glimpse/witness/behold` return arrays. See `docs/architecture.md` for the design spec and README for examples.

## Running Tests

```bash
uv run --extra dev python -m pytest tests/ -v
```

To generate reference images for render tests:

```bash
uv run python tests/generate_references.py
```

Two golden-image sets guard the renderers. `behold` covers the Mitsuba path
tracer; `soar` covers the witness/soar WGSL renderer via the eight frozen
judge views of the lighting loop, and so pins the browser's look too. The
soar set must be baked on real GPU hardware — the generator refuses a
software rasterizer rather than falling back to one, and the test skips when
no usable adapter is present. Re-bake one set with `--target soar` or
`--target behold`; see `tests/conftest.py` for the views and for how the
tolerances were measured.

## Data Files

Test data is stored in `data/`:
- `TWPICE_subvolume_256x256_5km.nc` - Real cloud data from TWPICE campaign
- `QC_FIF_Square_512,512,256.nc` - Synthetic FIF cloud field
