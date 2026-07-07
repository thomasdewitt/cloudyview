"""Camera: shared viewpoint description for the render functions.

Conventions (see config.py):
- Coordinates are meteorological: +x east, +y north, +z up.
- `position` is in relative coordinates: in x and y, ±1.0 reaches the
  domain edge. In z the renderers anchor -1 to the physical surface
  (z = 0) and +1 to the top of the data domain (witness), or span the
  domain height (behold).
- `azimuth` is a meteorological bearing in degrees: 0 north, 90 east,
  180 south, 270 west (clockwise from north).
- `elevation` is degrees above the horizon (0 horizon, 90 zenith).
- `fov` is the vertical field of view in degrees.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from . import config
from .angles import direction_from_azimuth_elevation

_DEFAULT_CAMERA = config.DEFAULT_WITNESS_CONFIG['camera']


@dataclass
class Camera:
    """Viewpoint in CloudyView's relative-coordinate convention."""

    position: Tuple[float, float, float] = field(
        default_factory=lambda: tuple(_DEFAULT_CAMERA['position'])
    )
    azimuth: float = _DEFAULT_CAMERA['azimuth']
    elevation: float = _DEFAULT_CAMERA['elevation']
    fov: float = _DEFAULT_CAMERA['fov']

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
        """Orthonormal (forward, right, up) camera basis.

        Matches the construction used by the witness renderer: world-up
        unless the view is within ~2.5 degrees of vertical, in which case
        +y is used as the up reference.
        """
        forward = self.forward
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return forward, right, up
