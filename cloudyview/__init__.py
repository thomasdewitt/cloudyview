"""CloudyView: 3D cloud field visualization toolkit.

Library usage:

    import cloudyview as cv

    field = cv.load("cloud.nc")                        # -> CloudField
    cam = cv.Camera(position=(0, -0.8, -0.95), azimuth=0, elevation=35)

    albedo = cv.glimpse(field)                         # (ny, nx) array
    img = cv.witness(field, camera=cam, size=(600, 400))   # (H, W, 3) array
    img = cv.behold(field, camera=cam, quality="high")     # (H, W, 3) array

Note: the public render functions shadow the same-named submodules on the
package (``cv.glimpse`` is the function). The submodules stay importable
directly, e.g. ``from cloudyview.witness import NestedLevel``.
"""

__version__ = "0.1.0"

from . import io
from . import basic_render
from . import optical_depth
from . import domain
from .optical_depth import vertically_integrated_optical_depth
from .domain import DomainGeometry, compute_domain_geometry
from .cloudfield import CloudField, load
from .camera import Camera

# Public render functions. These intentionally shadow the glimpse/witness/
# behold submodule attributes on the package (see module docstring).
from .glimpse import glimpse

__all__ = ["io", "basic_render", "optical_depth", "domain",
           "vertically_integrated_optical_depth",
           "DomainGeometry", "compute_domain_geometry",
           "CloudField", "load", "Camera",
           "glimpse"]
