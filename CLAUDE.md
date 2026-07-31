# CLAUDE.md - CloudyView Project Instructions

## Python Environment

This project uses **uv** for dependency management. A `.venv` and `uv.lock` are already present.

When running Python commands via Bash, use `uv run`:

```bash
uv run python your_command_here
uv run witness cloud.nc
```

For commands that need optional dependencies (tests, interactive engine):

```bash
uv run --extra dev python -m pytest tests/ -v
uv run --extra interactive python -m cloudyview.soar cloud.nc
```

## Project Overview

CloudyView is a 3D cloud field visualization toolkit with radiative transfer capabilities. It uses Mitsuba 3 for Monte Carlo path tracing with physically-based sky models and Mie scattering phase functions.

## Key Commands

- `behold <file.nc> --cpu|--gpu [quality]` - Photorealistic path-traced render (Mitsuba 3)
- `glimpse <file.nc>` - Quick 2D visual-albedo top view
- `witness <file.nc>` - Fast volumetric ray-marched render (numba CPU — the golden look reference)
- `soar <file.nc>` / `python -m cloudyview.soar` - Real-time WGSL fly-through (needs `--extra interactive`).
  `--nest <fine.nc>` adds a second, finer field nested inside the main one (placed by its own
  absolute coordinates; must lie inside the outer domain or it raises).

Library API (preferred): `cv.load()` → `CloudField`, `cv.Camera`, `cv.glimpse/witness/behold` return arrays. See `docs/architecture.md` for the design spec and README for examples.

## Running Tests

```bash
uv run --extra dev python -m pytest tests/ -v
```

To generate reference images for render tests:

```bash
uv run python tests/generate_references.py
```

## Data Files

Test data is stored in `data/`:
- `TWPICE_subvolume_256x256_5km.nc` - Real cloud data from TWPICE campaign
- `QC_FIF_Square_512,512,256.nc` - Synthetic FIF cloud field
