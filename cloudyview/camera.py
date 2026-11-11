"""Camera: the viewpoint the render functions share, and its angle conventions.

- Coordinates are meteorological: +x east, +y north, +z up.
- `position` is in relative coordinates: in x and y, +/-1.0 reaches the
  domain edge. In z the renderers anchor -1 to the physical surface
  (z = 0) and +1 to the top of the data domain.
- `azimuth` is a meteorological bearing in degrees: 0 north, 90 east,
  180 south, 270 west (clockwise from north). Sun angles use the same
  convention.
- `elevation` is degrees above the horizon (0 horizon, 90 zenith, -90 nadir).
- `fov` is the HORIZONTAL field of view in degrees, in all three renderers
  (behold's is Mitsuba's `fov_axis="x"`). The vertical half-angle is
  atan(tan(fov/2) / aspect).
"""

from dataclasses import dataclass
from typing import Tuple

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

    (soar/uniforms.py has its own copy written with math.cos and math.sin,
    which is the one held byte for byte against the browser; see there.)
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


@dataclass
class Camera:
    """Viewpoint in CloudyView's relative-coordinate convention.

    The default looks north from just above the surface at the domain's
    centre, 35 degrees above the horizon.
    """

    position: Tuple[float, float, float] = (0.0, 0.0, -0.999)
    azimuth: float = 0.0
    elevation: float = 35.0
    fov: float = 100.0

    def __post_init__(self):
        position = tuple(float(p) for p in self.position)
        if len(position) != 3:
            raise ValueError(
                f"position must be (x, y, z); got {len(position)} values."
            )
        self.position = position
        self.azimuth = float(self.azimuth)
        self.elevation = float(self.elevation)
        self.fov = float(self.fov)
        if not -90.0 <= self.elevation <= 90.0:
            raise ValueError(
                f"elevation must be in [-90, 90] degrees; got {self.elevation}."
            )
        if not 0.0 < self.fov < 180.0:
            raise ValueError(
                f"fov must be in (0, 180) degrees; got {self.fov}."
            )

    @property
    def forward(self) -> np.ndarray:
        """Unit view direction [x east, y north, z up]."""
        return direction_from_azimuth_elevation(self.azimuth, self.elevation)

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal (forward, right, up) camera basis (no roll).

        The right vector of a yaw/pitch camera is horizontal and depends
        only on azimuth: normalize(cross(forward, world_up)) reduces
        analytically to (cos az, -sin az, 0). The closed form keeps the
        basis continuous through straight up and down, where the cross
        product degenerates and would snap the view over.
        """
        forward = self.forward
        az = np.deg2rad(self.azimuth)
        right = np.array([np.cos(az), -np.sin(az), 0.0])
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return forward, right, up
