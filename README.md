# CloudyView

A Python toolkit for 3D cloud field visualization with optical depth calculations and radiative transfer modeling.

## Overview

CloudyView provides three tiers of visualization capabilities for 3D cloud condensate fields (from LES, cloud-resolving models, or other sources):

- **Glimpse** (`glimpse`): Quick optical depth calculation + matplotlib 3D visualization
- **Expedition** (`expedition`): Optical depth + simple 3D radiative transfer (two-stream)
- **Odyssey** (`odyssey`): Full 3D radiative transfer with comprehensive diagnostics

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

### Glimpse: Quick Look

```bash
glimpse example_cloud.nc
```

Outputs:
- 3D isosurface plot (qn ≥ 0.01 g/kg)
- 2D slice views (x-y, y-z, x-z)

### Expedition: Simple Radiative Transfer

```bash
expedition example_cloud.nc --sza 30
```

Outputs everything from Glimpse plus:
- Optical depth slices
- Reflectance field
- Transmission field

### Odyssey: Full Analysis

```bash
odyssey example_cloud.nc --output ./results --sza 30 --wavelength 0.55
```

Outputs everything from Expedition plus:
- Cloud water slices
- Heating rate calculations
- Summary statistics

## Input Data Requirements

NetCDF files must contain:
- **Liquid water variable** (required): One of `qc`, `ql`, `LWC`, `cloud_liquid_water_mixing_ratio`, `liquid_water_content`, `q_liquid`, or `lwc`
- **Ice water variable** (optional): One of `qi`, `qice`, `IWC`, `cloud_ice_mixing_ratio`, `ice_water_content`, `q_ice`, or `iwc`
- **Spatial dimensions**: Must be 3D (e.g., x, y, z or lon, lat, height)
- **Temporal dimension**: Must contain exactly one timestep

## Usage Examples

### Display plots interactively

```bash
# Quick look
glimpse my_cloud_data.nc

# With simple RT
expedition my_cloud_data.nc

# Full analysis
odyssey my_cloud_data.nc
```

### Save plots to files

```bash
# Save to output directory
glimpse my_cloud_data.nc --output ./plots

expedition my_cloud_data.nc --output ./plots --sza 45

odyssey my_cloud_data.nc --output ./plots --sza 45 --wavelength 0.55
```

### Specify solar geometry

```bash
# Different solar zenith angles
expedition cloud.nc --sza 0    # Sun overhead
expedition cloud.nc --sza 45   # Mid-morning/afternoon
expedition cloud.nc --sza 80   # Near horizon
```

### Different wavelengths (Odyssey only)

```bash
# Visible light
odyssey cloud.nc --wavelength 0.55

# Near-infrared
odyssey cloud.nc --wavelength 1.6

# Thermal infrared
odyssey cloud.nc --wavelength 11.0
```

## Module Structure

### Core Modules

- **`io.py`**: NetCDF file handling and data validation
  - `load_data()`: Load NetCDF file with xarray
  - `infer_liquid_water()`: Auto-detect liquid water variable
  - `infer_ice_water()`: Auto-detect ice water variable
  - `load_and_validate()`: Complete data loading with validation

- **`basic_render.py`**: Matplotlib-based visualization
  - `plot_isosurface()`: 3D isosurface at specified threshold
  - `plot_slices()`: 2D slice views along three dimensions

- **`optical_depth.py`**: Optical depth calculations
  - `calculate_optical_depth()`: Compute optical depth from water content
  - `column_optical_depth()`: Vertically integrate to column values

- **`radiative_transfer.py`**: Radiative transfer modeling
  - `simple_3d_radiative_transfer()`: Two-stream approximation
  - `advanced_3d_radiative_transfer()`: Full 3D RT with multiple scattering

### CLI Scripts

- **`glimpse.py`**: Entry point for quick visualization
- **`expedition.py`**: Entry point for radiative transfer analysis
- **`odyssey.py`**: Entry point for comprehensive visualization

## Features

### Visualizations
- 3D isosurface plots with configurable thresholds
- 2D slice views for detailed field inspection
- Multiple radiative transfer diagnostics
- Small figure sizes optimized for 13" MacBook displays

### Data Handling
- Automatic variable name inference (tries common naming conventions)
- Comprehensive validation (checks dimensions, timestep count)
- Support for both liquid and ice water phases
- Works with different coordinate systems

### Radiative Transfer
- Placeholder implementations ready for your custom code
- Two-stream approximation (Expedition)
- Full 3D radiative transfer framework (Odyssey)
- Configurable solar geometry and wavelength

## Customization

To integrate your own optical depth or radiative transfer code:

1. **Optical depth**: Modify `cloudyview/optical_depth.py`
   - Replace `calculate_optical_depth()` function
   - Maintain input/output array types (xarray)

2. **Radiative transfer**: Modify `cloudyview/radiative_transfer.py`
   - Replace `simple_3d_radiative_transfer()` and/or `advanced_3d_radiative_transfer()`
   - Return dictionary with result fields

3. **Visualization**: Modify `cloudyview/basic_render.py`
   - Add new visualization functions
   - Import and use in CLI scripts

## Dependencies

- `numpy`: Array operations
- `matplotlib`: Plotting and visualization
- `xarray`: NetCDF file handling with labeled arrays
- `netCDF4`: NetCDF4 file support (required by xarray for many operations)

## Mie Phase Function Tables

CloudyView includes pre-computed Mie scattering phase function tables for accurate cloud droplet scattering:

- **Mie0.txt**: Forward scattering peak (strongly forward-scattered component)
- **MiePF3.txt**: Rest of phase function distribution (sideways and backward scattering, wavelength-dependent RGB values)

These tables are from the PhD thesis:

**Antoine Bouthors (2008)** - "Real-time realistic rendering of clouds"
PhD Thesis, Université Grenoble I - Joseph Fourier
https://theses.hal.science/tel-00319974

The dual-table approach enables efficient importance sampling by separately handling the extremely strong forward peak and the much weaker sideways/backward scattering regions.

### Rendering Modes

CloudyView supports three rendering modes with different tradeoffs:

#### 1. **RGB Mode** (default) - `--mode rgb`
- Single RGB render with wavelength-averaged Mie phase function
- Full physically-based sunsky with realistic colors
- Best for general visualization and aesthetic renders
- Computational cost: 1× (baseline)

#### 2. **Mono Mode** - `--mode mono`
- Single monochrome render, grayscale output
- Fastest option for quick previews
- Computational cost: ~0.33× (3× faster than RGB)

#### 3. **Chromatic Mode** - `--mode chromatic`
- Renders 3 monochrome channels (R, G, B) separately
- Each channel uses wavelength-specific Mie phase function
- Enables chromatic scattering effects: coronas, halos, glories, rainbows
- Uses simplified constant sky (sunsky incompatible with monochrome)
- Computational cost: ~1× (3 mono renders ≈ 1 RGB render)

**Example usage:**

```bash
# Default RGB mode with realistic sky
appreciate cloud.nc high

# Chromatic mode for coronas/halos (looking toward sun)
appreciate cloud.nc high --mode chromatic --camera-azimuth 90 --camera-elevation 30

# Fast mono preview
appreciate cloud.nc low --mode mono
```

**In Python:**

```python
view_config = {
    'render_mode': 'chromatic',  # or 'mono' or 'rgb'
    # ... other config parameters
}
```

## Author

Thomas D. DeWitt (https://github.com/thomasdewitt/)

## License

MIT License - See LICENSE file for details
