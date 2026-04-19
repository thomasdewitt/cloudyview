"""
Configuration system for CloudyView

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction

Camera positions use relative coordinates where ±1.0 = domain edge in x,y
(z coordinates are relative to domain height)

Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
         (measured clockwise from +y axis; meteorological bearing)
Elevation: Angle above horizon (0° = horizon, 90° = zenith, -90° = nadir)

Sun angles use same convention.
"""

from pathlib import Path
from typing import Dict, Any, Optional


# Default configuration for witness (tier 2 - volume ray marching)
DEFAULT_WITNESS_CONFIG = {
    'camera': {
        'position': [0, 0, -0.999],  # x,y,z in relative coords (±1.0 = domain edge)
        'azimuth': 0.0,  # 0=North, 90=East
        'elevation': 35.0,  # angle above horizon
        'fov': 100.0,  # field of view in degrees
    },
    'sun': {
        'azimuth': 20.0,  # 0=North, 90=East
        'elevation': 55.0,  # degrees above horizon
    },
    'rendering': {
        'width': 600,
        'height': 400,
        'n_light_steps': 512,
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
        'position': [0, 0, -0.999], # x,y,z
        'azimuth': 0.0,
        'elevation': 35.0, 
        'fov': 100.0,  # Field of view in degrees
    },
    'sun': {
        'azimuth': 20.0,
        'elevation': 55.0,  # 45° above horizon
    },
    'rendering': {
        'max_depth': 128,
        'rr_depth': 64,
        'exposure': 4.0,
        'extinction_multiplier': 1.0,
        'integrator': 'volpathmis',
        'turbidity': 2.0,
        'ground_albedo': 0.5,
        'ocean': {
            'enabled': True,
            'reflectance': [0.0020, 0.0045, 0.0126],  # Matches witness OCEAN_REFLECTANCE
            'height': -0.9999,  # Just above ground plane
        },
    }
}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load built-in default configuration.

    External config files are intentionally disabled. Runtime customization
    should be done with CLI arguments only.

    Returns
    -------
    dict
        Configuration dictionary with 'witness' and 'behold' keys
    """
    if config_path is not None:
        raise NotImplementedError(
            "Config files are not supported. Use built-in defaults plus CLI overrides."
        )
    return {
        'witness': _deep_copy(DEFAULT_WITNESS_CONFIG),
        'behold': _deep_copy(DEFAULT_BEHOLD_CONFIG),
    }


def _deep_copy(obj):
    """Deep copy a nested dict/list structure"""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    else:
        return obj


def get_witness_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get configuration for witness (tier 2)

    Parameters
    ----------
    config_path : Path, optional
        Not supported. Must be None.

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
        Not supported. Must be None.

    Returns
    -------
    dict
        Behold configuration
    """
    return load_config(config_path)['behold']
