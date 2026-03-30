# CLAUDE.md - CloudyView Project Instructions

## Python Environment

**IMPORTANT**: This project requires a specific conda environment:

```bash
conda activate cloud-vis
```

When running Python commands via Bash, always use:

```bash
conda activate cloud-vis && python your_command_here
```

The bash shell resets between commands, so conda environment activation must be included in each command that requires Python packages.

## Project Overview

CloudyView is a 3D cloud field visualization toolkit with radiative transfer capabilities. It uses Mitsuba 3 for Monte Carlo path tracing with physically-based sky models and Mie scattering phase functions.

## Key Commands

- `behold <file.nc> --cpu|--gpu [quality]` - Photorealistic path-traced render (Mitsuba 3)
- `glimpse <file.nc>` - Quick 2D optical depth visualization
- `witness <file.nc>` - Fast volumetric ray-marched render

## Running Tests

```bash
conda activate cloud-vis && pytest tests/ -v
```

To generate reference images for render tests:

```bash
conda activate cloud-vis && python tests/generate_references.py
```

## Data Files

Test data is stored in `data/`:
- `TWPICE_subvolume_256x256_5km.nc` - Real cloud data from TWPICE campaign
- `QC_FIF_Square_512,512,256.nc` - Synthetic FIF cloud field
