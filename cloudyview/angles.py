"""Angle conversion helpers for CloudyView coordinate conventions."""

import numpy as np


def normalize_azimuth_deg(azimuth_deg: float) -> float:
    """Normalize azimuth to [0, 360)."""
    return float(azimuth_deg) % 360.0


def azimuth_met_to_internal_deg(azimuth_deg: float) -> float:
    """
    Convert meteorological azimuth to internal math azimuth.

    Input/output conventions:
    - Meteorological (input): 0°=North, 90°=East, clockwise
    - Internal (output): 0°=East(+x), 90°=North(+y), counterclockwise
    """
    return (90.0 - normalize_azimuth_deg(azimuth_deg)) % 360.0


def direction_from_azimuth_elevation(
    azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    """
    Build a unit direction vector [x East, y North, z Up].

    Azimuth uses meteorological convention: 0°=North, 90°=East, clockwise.
    Elevation is angle above horizon.
    """
    az_internal_rad = np.deg2rad(azimuth_met_to_internal_deg(azimuth_deg))
    el_rad = np.deg2rad(elevation_deg)
    cos_el = np.cos(el_rad)
    direction = np.array(
        [
            cos_el * np.cos(az_internal_rad),
            cos_el * np.sin(az_internal_rad),
            np.sin(el_rad),
        ],
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    return direction
