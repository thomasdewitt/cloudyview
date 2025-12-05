# CloudyView

A Python toolkit for 3D cloud field visualization with optical depth calculations and Monte Carlo radiative transfer.

## Overview

CloudyView provides three tiers of visualization capabilities for 3D cloud condensate fields (from LES, cloud-resolving models, or other sources):

- **Glimpse** (`glimpse`): Quick optical depth calculation + matplotlib 2D visualization
- **Witness** (`witness`): PyVista 3D isosurface rendering with configurable views
- **Behold** (`behold`): Photorealistic Monte Carlo path tracing with Mitsuba 3

## Coordinate System

CloudyView uses the **meteorological convention**:
- **East** = +x direction
- **North** = +y direction
- **Up** = +z direction

Azimuth angles are measured counterclockwise from East:
- 0° = East, 90° = North, 180° = West, 270° = South

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

## Quick Start

All scripts require a NetCDF file with a cloud water mixing ratio variable and are designed for single-timestep 3D data.

### Glimpse: Quick 2D View

```bash
glimpse example_cloud.nc
```

Generates a matplotlib visualization of column optical depth.

### Witness: 3D Isosurface Views

```bash
witness example_cloud.nc
```

Generates multiple 3D views (overhead, north oblique, west oblique) using PyVista isosurface rendering.

Options:
- `--interactive`: Export interactive HTML instead of PNG
- `-n 10`: Number of isosurfaces (default: 10)
- `--min-threshold 0.001 --max-threshold 1.0`: Optical depth range

### Behold: Photorealistic Rendering

```bash
behold example_cloud.nc llvm medium
```

Generates photorealistic path-traced render using Mitsuba 3.

Arguments:
- `backend`: `llvm` (CPU) or `cuda` (GPU) - **required**
- `quality`: `min`, `low`, `medium` (default), or `high`

Quality tiers:
- `min`: 200×400, spp=1 (instant preview)
- `low`: 400×200, spp=32 (quick preview)
- `medium`: 800×400, spp=512 (balanced, default)
- `high`: 1600×800, spp=4096 (production quality)

Options:
- `--output ./renders`: Output directory (default: current directory)

## Input Data Requirements

NetCDF files must contain:
- **Liquid water variable** (required): One of `qc`, `ql`, `LWC`, `cloud_liquid_water_mixing_ratio`, `liquid_water_content`, `q_liquid`, or `lwc`
- **Ice water variable** (optional): One of `qi`, `qice`, `IWC`, `cloud_ice_mixing_ratio`, `ice_water_content`, `q_ice`, or `iwc`
- **Spatial dimensions**: Must be 3D (e.g., x, y, z or lon, lat, height)
- **Temporal dimension**: Must contain exactly one timestep

## Configuration

Witness and Behold can be configured via YAML files. Settings include camera positions, sun angles, rendering parameters, and more.

### Configuration File Locations

CloudyView searches for configuration files in this order:
1. `./cloudyview.yaml` (current directory)
2. `~/.cloudyview.yaml` (home directory)
3. Built-in defaults if no config file found

### Creating a Configuration File

Copy the example configuration:

```bash
cp cloudyview.yaml.example cloudyview.yaml
```

Then edit `cloudyview.yaml` to customize:
- Camera positions (relative coordinates where ±1.0 = domain edge)
- Sun azimuth and elevation
- Rendering parameters (max depth, exposure, etc.)
- Ocean surface settings

See `cloudyview.yaml.example` for full documentation of all options.

## Usage Examples

### Quick Visualization Pipeline

```bash
# 1. Quick 2D overview
glimpse my_cloud_data.nc

# 2. 3D isosurface views
witness my_cloud_data.nc

# 3. Photorealistic render
behold my_cloud_data.nc cuda high --output ./renders
```

### Witness Examples

```bash
# Default views (overhead, north, west)
witness cloud.nc

# Interactive HTML export
witness cloud.nc --interactive

# Single high optical depth isosurface
witness cloud.nc --threshold 1.0

# Custom threshold range with more surfaces
witness cloud.nc -n 20 --min-threshold 0.0001 --max-threshold 10.0
```

### Behold Examples

```bash
# Fast CPU preview
behold cloud.nc llvm min

# Balanced GPU render
behold cloud.nc cuda medium

# Production quality render
behold cloud.nc cuda high --output ./final_renders

# Quick GPU preview
behold cloud.nc cuda low
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

- **`config.py`**: Configuration system
  - `load_config()`: Load configuration from YAML file
  - `get_witness_config()`: Get witness configuration
  - `get_behold_config()`: Get behold configuration

- **`basic_render.py`**: Matplotlib-based visualization
  - Column optical depth visualization

- **`radiative_transfer.py`**: Mitsuba 3 Monte Carlo path tracing
  - `load_mie_phase_tables()`: Load Mie scattering phase functions
  - `render_view()`: Render a single view with Mitsuba
  - `look_at_world_up()`: Camera transform helper
  - Support for RGB rendering with physically-based sky

### CLI Scripts

- **`glimpse.py`**: Entry point for quick 2D visualization
- **`witness.py`**: Entry point for PyVista 3D isosurface rendering
- **`behold.py`**: Entry point for Mitsuba photorealistic rendering

## Features

### Visualizations
- **Glimpse**: 2D column optical depth visualization
- **Witness**: Multiple 3D isosurface views with physically-based opacity
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
- Configurable camera and sun positions via YAML
- Multiple quality levels from instant preview to production
- Progressive rendering with checkpoints
- Ocean surface with realistic reflections

### Configuration
- YAML-based configuration system
- Relative coordinate positioning
- Multiple camera views for witness
- Customizable rendering parameters

## Dependencies

### Core Dependencies
- `numpy>=1.20`: Array operations
- `matplotlib>=3.3`: 2D plotting
- `xarray>=0.18`: NetCDF file handling with labeled arrays
- `netCDF4>=1.5`: NetCDF4 file support
- `pyyaml>=5.4`: Configuration file parsing

### Optional Dependencies
- **For witness (3D isosurfaces)**: `pyvista>=0.37.0`
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
