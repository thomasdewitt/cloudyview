# CloudyView

A Python toolkit for 3D cloud field visualization with optical depth calculations and Monte Carlo radiative transfer.

## Overview

CloudyView provides three tiers of visualization capabilities for 3D cloud condensate fields (from LES, cloud-resolving models, or other sources):

- **Glimpse** (`glimpse`): Quick optical depth calculation + matplotlib 2D visualization
- **Witness** (`witness`): Fast volumetric ray marching with multi-scattering approximation
- **Behold** (`behold`): Photorealistic Monte Carlo path tracing with Mitsuba 3

## Coordinate System

CloudyView uses the **meteorological convention**:

- **East** = +x direction
- **North** = +y direction
- **Up** = +z direction

Azimuth angles use the meteorological convention (clockwise from North):

- 0° = North, 90° = East, 180° = South, 270° = West

Elevation angles are measured from the horizon:

- 0° = horizon, 90° = zenith, -90° = nadir

## Installation

### From source (development)

```bash
cd /path/to/cloudyview
pip install -e .
```

### With optional development tools

```bash
pip install -e ".[dev]"
```

## Library usage

CloudyView is library-first: 3D cloud volume in, image array out. The CLIs
below are thin wrappers over these functions.

```python
import cloudyview as cv

# Load — one file with both variables, or split files (SAM LPT-style
# output writes one variable per file: ..._QC_*.nc, ..._QI_*.nc)
field = cv.load("cloud.nc")                          # autodetect qc + qi
field = cv.load("..._QC_0000000600.nc",
                ice="..._QI_0000000600.nc")          # split liquid/ice files
field = cv.load("cloud.nc", liquid_water_var="QC")   # explicit overrides

field.lwc, field.iwc      # (nx, ny, nz) float32 g/kg; iwc may be None
field.x, field.y, field.z # 1D coords, meters

cam = cv.Camera(position=(0, -0.8, -0.95),  # relative coords, ±1 = domain edge
                azimuth=0,                  # met bearing: 0=N, 90=E
                elevation=35,               # degrees above horizon
                fov=100)                    # vertical field of view

albedo = cv.glimpse(field)                        # (ny, nx) two-stream visual albedo
img = cv.witness(field, camera=cam, size=(600, 400))   # (H, W, 3) in [0, 1]
img = cv.behold(field, camera=cam, quality="high")     # (H, W, 3), Mitsuba 3

cv.save_image(img, "render.png")
```

Notes:

- Render functions return arrays and never write files; use `cv.save_image`
  or matplotlib for output.
- `cv.witness(...)` uses the maintained numba CPU ray marcher.
  `cv.behold(..., gpu=True)` selects Mitsuba's CUDA variant.
- Library code raises exceptions and is quiet by default
  (`verbose=True` restores the CLI-style diagnostics).
- The functions `cv.glimpse` / `cv.witness` / `cv.behold` shadow the
  same-named submodules on the package namespace; the modules remain
  importable directly (e.g. `from cloudyview.witness import NestedLevel`).

## Quick Start

All scripts require a NetCDF file with a cloud water mixing ratio variable and are designed for single-timestep 3D data.

### Glimpse: Quick 2D View

```bash
glimpse example_cloud.nc
```

Generates a matplotlib visualization of column optical depth.

### Witness: Fast Volumetric Rendering

```bash
witness example_cloud.nc
```

Generates a volumetric ray-marched render with multi-scattering, procedural sky, and ocean surface.

Options:

- `--camera-position X Y Z`: Camera position in relative coords (default: 0 0 -0.999)
- `--camera-azimuth`, `--camera-elevation`, `--fov`: Camera orientation
- `--sun-azimuth`, `--sun-elevation`: Sun position
- `--size W H`: Image dimensions
- Quality presets: `min`, `low`, `medium` (default), `high`

### Behold: Photorealistic Rendering

```bash
behold example_cloud.nc --cpu
```

Generates photorealistic path-traced render using Mitsuba 3.

Arguments:

- `--cpu` or `--gpu`: Backend selection (required)
- `quality`: `min`, `low`, `medium` (default), `high`, or `custom`

Quality tiers:

- `min`: 150×100, spp=1, max_depth=4, rr_depth=2 (instant preview)
- `low`: 300×200, spp=32, max_depth=16, rr_depth=4 (quick preview)
- `medium`: 600×400, spp=512, max_depth=64, rr_depth=16 (balanced, default)
- `high`: 1200×800, spp=4096, max_depth=128, rr_depth=32 (production quality)
- `custom`: User-specified spp, resolution, and/or max_depth/rr_depth

Options:

| Argument                  | Default        | Description                                              |
| ------------------------- | -------------- | -------------------------------------------------------- |
| `--output`, `-o`          | `.`            | Output directory for renders                             |
| `--spp N`                 | varies         | Samples per pixel (for custom quality)                   |
| `--size W H`              | varies         | Image dimensions in pixels (for custom quality)          |
| `--max-depth N`           | varies         | Maximum ray bounce depth (for custom quality override)   |
| `--rr-depth N`            | varies         | Russian roulette depth (for custom quality override)     |
| `--camera-position X Y Z` | `0 0 -0.999`   | Camera position in relative coords (±1.0 = domain edge)  |
| `--camera-azimuth`        | `0`            | Camera view azimuth in degrees (0=N, 90=E, 180=S, 270=W) |
| `--camera-elevation`      | `35`           | Camera view elevation in degrees (angle above horizon)   |
| `--fov`                   | `100`          | Camera field of view in degrees                          |
| `--sun-azimuth`           | `20`           | Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)         |
| `--sun-elevation`         | `55`           | Sun elevation in degrees (angle above horizon)           |

## Input Data Requirements

NetCDF files must contain:

- **Liquid water variable** (required): One of `qc`, `ql`, `LWC`, `cloud_liquid_water_mixing_ratio`, `liquid_water_content`, `q_liquid`, or `lwc`
- **Ice water variable** (optional): One of `qi`, `qice`, `IWC`, `cloud_ice_mixing_ratio`, `ice_water_content`, `q_ice`, or `iwc`
- **Spatial dimensions**: Must be 3D (e.g., x, y, z or lon, lat, height)
- **Temporal dimension**: Must contain exactly one timestep

## Usage Examples

### Quick Visualization Pipeline

```bash
# 1. Quick 2D overview
glimpse my_cloud_data.nc

# 2. Fast volumetric render
witness my_cloud_data.nc

# 3. Photorealistic render
behold my_cloud_data.nc high --gpu --output ./renders
```

### Witness Examples

```bash
# Default render
witness cloud.nc

# High quality with custom camera
witness cloud.nc high --camera-position 0 -0.9 -0.99 --camera-azimuth 0 --camera-elevation 35

# Custom size
witness cloud.nc medium --size 1200 800
```

### Behold Examples

```bash
# Fast CPU preview
behold cloud.nc --cpu

# Balanced GPU render
behold cloud.nc --gpu

# Production quality render
behold cloud.nc high --gpu --output ./final_renders

# Quick GPU preview
behold cloud.nc low --gpu

# Custom quality: 1024x768 at 256 spp with max_depth=64
behold cloud.nc custom --gpu --size 1024 768 --spp 256 --max-depth 64 --rr-depth 32
```

## Module Structure

### Core Modules

- **`io.py`**: NetCDF file handling and data validation
  
  - `load_data()`: Load NetCDF file with xarray
  - `infer_liquid_water()`: Auto-detect liquid water variable
  - `infer_ice_water()`: Auto-detect ice water variable
  - `load_and_validate()`: Complete data loading with validation

- **`optical_depth.py`**: Optical depth calculations
  
  - `compute_extinction_field()`: Compute extinction coefficient from water content
  - Supports variable vertical spacing

- **`domain.py`**: Shared domain geometry

  - `DomainGeometry`: Dataclass with physical dimensions and aspect ratios
  - `compute_domain_geometry()`: Factory from coordinate arrays (handles non-uniform dz)

- **`config.py`**: Built-in configuration defaults

  - `get_witness_config()`: Get witness configuration
  - `get_behold_config()`: Get behold configuration

- **`basic_render.py`**: Matplotlib-based visualization

  - Column optical depth visualization

- **`radiative_transfer.py`**: Mitsuba 3 Monte Carlo path tracing

  - `load_mie_phase_tables()`: Load Mie scattering phase functions
  - `render_view()`: Render a single RGB view with Mitsuba
  - `look_at_world_up()`: Camera transform helper
  - Physically-based Preetham sunsky model

### CLI Scripts

- **`glimpse.py`**: Entry point for quick 2D visualization
- **`witness.py`**: Entry point for volumetric ray marching
- **`behold.py`**: Entry point for Mitsuba photorealistic rendering

## Features

### Visualizations

- **Glimpse**: 2D column optical depth visualization
- **Witness**: Fast volumetric ray marching with multi-scattering
- **Behold**: Photorealistic Monte Carlo path tracing with Mitsuba 3

### Data Handling

- Automatic variable name inference (tries common naming conventions)
- Comprehensive validation (checks dimensions, timestep count)
- Support for both liquid and ice water phases
- Works with different coordinate systems
- Variable vertical spacing supported

### Radiative Transfer (Behold)

- Full volumetric path tracing with Mitsuba 3
- Physically-based Preetham sunsky model
- Accurate Mie scattering from pre-computed phase functions
- Configurable camera and sun positions via CLI flags
- Multiple quality levels from instant preview to production
- Progressive rendering with checkpoints
- Ocean surface with realistic reflections

### Configuration

- Built-in defaults with CLI overrides
- Relative coordinate positioning (±1.0 = domain edge)
- Customizable rendering parameters via CLI flags

## Dependencies

### Core Dependencies

- `numpy>=1.20`: Array operations
- `matplotlib>=3.3`: 2D plotting
- `xarray>=0.18`: NetCDF file handling with labeled arrays
- `netCDF4>=1.5`: NetCDF4 file support
### Optional Dependencies

- **For witness (volumetric rendering)**: `numba>=0.56`
- **For behold (photorealistic rendering)**: `mitsuba>=3.0.0`, `drjit>=0.3.0`

Install all optional dependencies:

```bash
pip install -e ".[all]"
```

## Mie Phase Function Tables

Behold uses pre-computed Mie scattering phase function tables for accurate cloud scattering.

### Liquid Water Droplets

- **Mie0_normalized.txt**: Forward scattering peak (strongly forward-scattered component)
- **MiePF3_normalized.txt**: Rest of phase function distribution (sideways and backward scattering, wavelength-dependent RGB values)

These tables are from the PhD thesis:

**Antoine Bouthors (2008)** - "Real-time realistic rendering of clouds"
PhD Thesis, Université Grenoble I - Joseph Fourier
https://theses.hal.science/tel-00319974

The dual-table approach enables efficient importance sampling by separately handling the extremely strong forward peak and the much weaker sideways/backward scattering regions. The tables are blended with equal weighting (0.5) for optimal sampling.

### Ice Crystals

- **IceMie0_normalized.txt**: Forward scattering peak for ice crystals (RGB wavelength-dependent)
- **IceMiePF3_normalized.txt**: Rest of phase function distribution for ice crystals (RGB wavelength-dependent)

These tables are derived from:

**Baum, B.A., Yang, P., Heymsfield, A.J., Bansemer, A., Cole, B.H., Merrelli, A., Schmitt, C., Wang, C. (2014)** - "Ice cloud single-scattering property models with the full phase matrix at wavelengths from 0.2 to 100 µm"
*Journal of Quantitative Spectroscopy and Radiative Transfer*, 146, 123-139.
https://www.sciencedirect.com/science/article/pii/S0022407314000867#ab0010

Full ice scattering database downloaded from: http://download.ssec.wisc.edu/files/polarization_models/

**Note**: The complete ice scattering data and processing scripts are located in `ice_models/`, which contains a script to reformat the original data and extract the phase functions needed for rendering.

## Author

Thomas D. DeWitt (https://github.com/thomasdewitt/)

## License

MIT License - See LICENSE file for details
