"""
Configuration system for CloudyView

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction

Camera positions use relative coordinates where ±1.0 = domain edge in x,y
(z coordinates are relative to domain height)

Azimuth: 0° = East, 90° = North, 180° = West, 270° = South
         (measured counterclockwise from +x axis)
Elevation: Angle above horizon (0° = horizon, 90° = zenith, -90° = nadir)

Sun angles use same convention.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Default configuration for witness (tier 2 - volume ray marching)
DEFAULT_WITNESS_CONFIG = {
    'camera': {
        'position': [0, 0, -0.9],  # x,y,z in relative coords (±1.0 = domain edge)
        'azimuth': 90.0,  # 0=East, 90=North
        'elevation': 35.0,  # angle above horizon
        'fov': 100.0,  # field of view in degrees
    },
    'sun': {
        'azimuth': 70.0,  # degrees from east
        'elevation': 55.0,  # degrees above horizon
    },
    'rendering': {
        'width': 600,
        'height': 400,
        'n_light_steps': 64,
        'exposure': 4.0,
        'extinction_multiplier': 1.0,
        'ocean': {
            'enabled': True,
            'reflectance': [0.0392, 0.1098, 0.1490],
            'height': -0.9999,
        },
    }
}


# Default configuration for behold (tier 3 - Mitsuba path tracing)
DEFAULT_BEHOLD_CONFIG = {
    'camera': {
        'position': [0, -.99, -0.5], # x,y,z
        'azimuth': 90.0, 
        'elevation': 35.0, 
        'fov': 100.0,  # Field of view in degrees
    },
    'sun': {
        'azimuth': 70.0, 
        'elevation': 55.0,  # 45° above horizon
    },
    'rendering': {
        'max_depth': 128,
        'rr_depth': 64,
        'exposure': 4.0,
        'extinction_multiplier': 1.0,
        'integrator': 'volpathmis',
        'turbidity': 3.0,
        'ground_albedo': 0.5,
        'ocean': {
            'enabled': True,
            'reflectance': [0.0392, 0.1098, 0.1490],  # Dark blue ocean
            'height': -0.9999,  # Just above ground plane
        },
    }
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Search order:
    1. Explicit path if provided
    2. ./cloudyview.yaml (current directory)
    3. ~/.cloudyview.yaml (home directory)
    4. Use defaults if no config file found

    Returns
    -------
    dict
        Configuration dictionary with 'witness' and 'behold' keys
    """
    # Start with defaults (deep copy)
    config = {
        'witness': _deep_copy(DEFAULT_WITNESS_CONFIG),
        'behold': _deep_copy(DEFAULT_BEHOLD_CONFIG)
    }

    # Search for config file
    if config_path is None:
        candidates = [
            Path('./cloudyview.yaml'),
            Path.home() / '.cloudyview.yaml'
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    # Load and merge if found
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)

                if user_config:
                    # Deep merge user config into defaults
                    if 'witness' in user_config:
                        _deep_merge(config['witness'], user_config['witness'])
                    if 'behold' in user_config:
                        _deep_merge(config['behold'], user_config['behold'])

                    print(f"  Loaded config from {config_path}")
            except Exception as e:
                print(f"  Warning: Failed to load config from {config_path}: {e}")
                print(f"  Using default configuration")

    return config


def _deep_copy(obj):
    """Deep copy a nested dict/list structure"""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    else:
        return obj


def _deep_merge(base: Dict, override: Dict) -> None:
    """
    Recursively merge override into base (modifies base in-place)

    Parameters
    ----------
    base : dict
        Base dictionary to merge into
    override : dict
        Override values to merge from
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_witness_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get configuration for witness (tier 2)

    Parameters
    ----------
    config_path : Path, optional
        Optional path to config file

    Returns
    -------
    dict
        Witness configuration
    """
    return load_config(config_path)['witness']


def get_behold_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get configuration for behold (tier 3)

    Parameters
    ----------
    config_path : Path, optional
        Optional path to config file

    Returns
    -------
    dict
        Behold configuration
    """
    return load_config(config_path)['behold']
