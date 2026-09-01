# CloudyView

A toolkit for 3D cloud field visualization with optical depth calculations and Monte Carlo radiative transfer.

## Interactive rendering in browser (recommended)

Visit [thomasddewitt.com/soar](https://thomasddewitt.com/soar), open your NetCDF file containing cloud condensate, adjust quality and time of day, and render images and videos. Soar uses the `witness` renderer, described below.

![Witness render of a STEAM cloud field, looking into the sun](https://raw.githubusercontent.com/thomasdewitt/cloudyview/master/docs/witness_small_c002_s0100_into_sun.png)

## Programmatic rendering: Overview

For rendering with more fine-grained control, `cloudyview` may be installed to give three terminal commands for renders in three increasing levels of cost:

- **Glimpse** (`glimpse`): Quick optical depth calculation + 2D visualization from above, like a satellite image
- **Witness** (`witness`): Fast volumetric ray marching with visually tuned heuristics, on GPU
- **Behold** (`behold`): Physically accurate Monte Carlo path tracing with Mitsuba 3. Very, very expensive for clouds.

The source code for Soar is in `web/soar`.

## Quick Start: Terminal Commands

For `witness` and `behold`, it is recommended to first choose your view in [Soar](https://thomasddewitt.com/soar), and use the "Render this view in terminal" command, which gives a command to copy-paste.

All scripts require a NetCDF file with cloud ice/water mixing ratio variables.

### Glimpse: Quick 2D View

```bash
glimpse example_cloud.nc
```

Generates a matplotlib visualization of column optical depth.

Options:

- `--output`, `-o`: Output directory (default: current directory)
- `--label-dirs`: Label N/S/W/E sections of the domain
- `--label`: Overlay camera marker and field-of-view on the top view
- `--camera-position X Y Z`, `--camera-azimuth`, `--camera-elevation`, `--fov`:
  Camera used by `--label` (same conventions as witness/behold)

### Witness: Fast Volumetric Rendering

```bash
witness example_cloud.nc
```

Generates a volumetric ray-marched render with multi-scattering, procedural sky, and ocean surface.

Quality presets (positional argument; image size only): `min` 150×100,
`low` 300×200, `medium` 600×400 (default), `high` 1600×1200.
`--size W H` overrides the preset.

Options:

| Argument                  | Default      | Description                                                                   |
| ------------------------- | ------------ | ----------------------------------------------------------------------------- |
| `--output`, `-o`          | `.`          | Output directory                                                              |
| `--camera-position X Y Z` | `0 0 -0.999` | Camera position in relative coords (±1.0 = domain edge)                       |
| `--camera-azimuth`        | `0`          | Camera azimuth in degrees (0=N, 90=E, 180=S, 270=W)                           |
| `--camera-elevation`      | `35`         | Camera elevation in degrees (angle above horizon)                             |
| `--fov`                   | `100`        | Horizontal field of view in degrees                                           |
| `--sun-azimuth`           | `20`         | Sun azimuth in degrees                                                        |
| `--sun-elevation`         | `55`         | Sun elevation in degrees                                                      |
| `--exposure`              | `4.0`        | Tone-map exposure (soar's "render in terminal" writes its metered value here) |
| `--gamma`                 | `1.66`       | Tone-map gamma                                                                |
| `--white-point`           | `15.0`       | Extended-Reinhard white point: the exposed radiance mapping to 1.0            |
| `--contrast`              | `1.0`        | Display contrast about mid-grey, applied after the gamma encode               |
| `--haze`                  | `1.0`        | Aerosol amount, 0 to 2                                                        |
| `--haze-height-dependent` | off          | Thin the haze with height on a 2500 m scale height                            |
| `--lod`                   | `0.5`        | Angular level of detail; smaller is finer and slower                          |
| `--periodic`              | off          | Wrap the domain in x and y, as soar does for LES fields                       |
| `--nest-group GROUP`      | —            | NetCDF group in the same file holding a finer field to render as a nest       |

The image controls are the same knobs the browser app exposes, so a soar
view reproduces exactly in the terminal. All three commands also accept the
dataset-selection overrides described under
[Input Data Requirements](#input-data-requirements) (`--group`,
`--liquid-water-var`, `--x-dim`, ...); see `witness --help` for the full
list.

### Behold: Photorealistic Rendering

```bash
behold example_cloud.nc --cpu
```

Generates photorealistic path-traced render using Mitsuba 3.

Arguments:

- `--cpu` or `--gpu`: Backend selection (required)
- `quality`: `min`, `low`, `medium` (default), `high`, `max`, or `custom`

Quality tiers:

- `min`: 150×100, spp=1, max_depth=2, rr_depth=1
- `low`: 300×200, spp=32, max_depth=4, rr_depth=2
- `medium`: 600×400, spp=512, max_depth=8, rr_depth=4 (default)
- `high`: 960×640, spp=1024, max_depth=8, rr_depth=4
- `max`: 1200×800, spp=2048, max_depth=96, rr_depth=64 (untruncated path depths)
- `custom`: User-specified spp, resolution, and/or max_depth/rr_depth
  (unspecified depths default to `max`'s)

Clouds bury most of a path tracer's budget in dense multiple scattering, so
`min` through `high` truncate paths early (`max_depth=8` or less) and accept
the darkening bias; `max` keeps the deep budgets and is not tractable for
most cloud fields.

Options:

| Argument                  | Default      | Description                                                           |
| ------------------------- | ------------ | --------------------------------------------------------------------- |
| `--output`, `-o`          | `.`          | Output directory for renders                                          |
| `--ice FILE`              | —            | Separate NetCDF file with the ice variable (SAM LPT split-file style) |
| `--spp N`                 | varies       | Samples per pixel (for custom quality)                                |
| `--size W H`              | varies       | Image dimensions in pixels (for custom quality)                       |
| `--max-depth N`           | varies       | Maximum ray bounce depth (for custom quality override)                |
| `--rr-depth N`            | varies       | Russian roulette depth (for custom quality override)                  |
| `--camera-position X Y Z` | `0 0 -0.999` | Camera position in relative coords (±1.0 = domain edge)               |
| `--camera-azimuth`        | `0`          | Camera view azimuth in degrees (0=N, 90=E, 180=S, 270=W)              |
| `--camera-elevation`      | `35`         | Camera view elevation in degrees (angle above horizon)                |
| `--fov`                   | `100`        | Camera field of view in degrees                                       |
| `--sun-azimuth`           | `20`         | Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)                      |
| `--sun-elevation`         | `55`         | Sun elevation in degrees (angle above horizon)                        |

## Installation for programmatic rendering

```bash
pip install cloudyview            # glimpse and witness
pip install 'cloudyview[behold]'  # + Mitsuba 3 for path-traced renders
```

`witness` needs a GPU (WebGPU via wgpu-py). The soar shader and ocean tiles
ship inside the package (`cloudyview/soar/`); the web app reaches the same
files through symlinks under `web/soar/`, so the browser and the Python host
always share one renderer core.

### Development install

```bash
git clone https://github.com/thomasdewitt/cloudyview
cd cloudyview
uv sync
```

or `pip install -e ".[dev]"`.

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
                fov=100)                    # horizontal field of view

albedo = cv.glimpse(field)                        # (ny, nx) two-stream visual albedo
img = cv.witness(field, camera=cam, size=(600, 400))   # (H, W, 3) in [0, 1]
img = cv.behold(field, camera=cam, quality="high")     # (H, W, 3), Mitsuba 3

cv.save_image(img, "render.png")
```

Notes:

- Render functions return arrays and never write files; use `cv.save_image`
  or matplotlib for output.
- `cv.witness(...)` drives soar's WGSL ray marcher through wgpu.
  `cv.behold(..., gpu=True)` selects Mitsuba's CUDA variant.
- Library code raises exceptions and is quiet by default
  (`verbose=True` restores the CLI-style diagnostics).
- The functions `cv.glimpse` / `cv.witness` / `cv.behold` shadow the
  same-named submodules on the package namespace; the modules remain
  importable directly (e.g. `from cloudyview.witness import NestedLevel`).

## Input Data Requirements

NetCDF files must contain:

- **Liquid water variable** (required): One of `qc`, `ql`, `LWC`, `cloud_liquid_water_mixing_ratio`, `liquid_water_content`, `q_liquid`, or `lwc`
- **Ice water variable** (optional): One of `qi`, `qice`, `IWC`, `cloud_ice_mixing_ratio`, `ice_water_content`, `q_ice`, or `iwc`
- **Spatial dimensions**: Must be 3D (e.g., x, y, z or lon, lat, height)
- **Temporal dimension**: Must contain exactly one timestep

When autodetection is not enough — non-standard names, variables inside
NetCDF groups, unusual dimension names — all three commands accept the same
override flags: `--group`, `--liquid-water-var`, `--liquid-water-group`,
`--ice-water-var`, `--ice-water-group`, `--coords-group`,
`--x-coord`/`--y-coord`/`--z-coord`, and `--x-dim`/`--y-dim`/`--z-dim`.
Physical x/y/z coordinates are always required (index axes are not enough:
optical depth and rendering depend on grid spacing). See `witness --help`
for the rules and example patterns.

## Coordinate System

CloudyView uses the **meteorological convention**:

- **East** = +x direction
- **North** = +y direction
- **Up** = +z direction

Azimuth angles use the meteorological convention (clockwise from North):

- 0° = North, 90° = East, 180° = South, 270° = West

Elevation angles are measured from the horizon:

- 0° = horizon, 90° = zenith, -90° = nadir

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

# Soar's max capture tier, custom camera
witness cloud.nc --quality max --camera-position 0 -0.9 -0.99 --camera-azimuth 0 --camera-elevation 35

# Custom size (default 600x400)
witness cloud.nc --size 1200 800
```

### Behold Examples

```bash
# Fast CPU preview
behold cloud.nc --cpu

# Balanced GPU render
behold cloud.nc --gpu

# Production quality render
behold cloud.nc max --gpu --output ./final_renders

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
- `wgpu>=0.30`: WebGPU for witness (needs a GPU at run time)

### Optional Dependencies

- **For behold (photorealistic rendering)**: `mitsuba>=3.0.0`, `drjit>=0.3.0` — `pip install 'cloudyview[behold]'`

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

**Note**: The complete ice scattering data and processing scripts are located in `data/ice_models/`, which contains a script to reformat the original data and extract the phase functions needed for rendering.

## Author

Thomas D. DeWitt (https://github.com/thomasdewitt/)

## License

MIT License - See LICENSE file for details
