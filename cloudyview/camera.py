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
- `fov` is the HORIZONTAL field of view in degrees. All three renderers
  agree on this: behold inherited it from Mitsuba's `fov_axis="x"` default,
  and witness and soar were brought onto it (2026-08-05) — before that they
  read the same number as vertical, so a camera handed from soar to behold
  came out zoomed in by the image aspect ratio. The vertical half-angle is
  atan(tan(fov/2) / aspect).
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
        """Orthonormal (forward, right, up) camera basis (no roll).

        The right vector of a yaw/pitch camera is horizontal and depends
        only on azimuth: normalize(cross(forward, world_up)) reduces
        analytically to (cos az, -sin az, 0). Using the closed form keeps
        the basis continuous through straight up/down — the historical
        cross-product construction flipped its up-reference within ~2.5
        degrees of vertical, visibly snapping the view (and the WASD
        frame) near the pole.
        """
        forward = self.forward
        az = np.deg2rad(self.azimuth)
        right = np.array([np.cos(az), -np.sin(az), 0.0])
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return forward, right, up
