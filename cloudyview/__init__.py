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

__version__ = "0.1.1"

from importlib import import_module
import sys
from types import ModuleType


def _ensure_drjit_llvm_path() -> None:
    """Point Dr.Jit at the system libLLVM when only a versioned .so exists.

    behold.py does the same. It is also done here because Dr.Jit reads
    DRJIT_LIBLLVM_PATH at the FIRST mitsuba import anywhere in the process,
    and importing cloudyview does not import behold, so behold's own call
    can come too late. It only looks at the environment and the filesystem.
    """
    import glob
    import os

    if os.environ.get("DRJIT_LIBLLVM_PATH"):
        return
    candidates = sorted(
        glob.glob("/usr/lib64/libLLVM.so*") + glob.glob("/usr/lib/libLLVM.so*")
        + glob.glob("/usr/lib/x86_64-linux-gnu/libLLVM*.so*")
    )
    if candidates:
        os.environ["DRJIT_LIBLLVM_PATH"] = candidates[-1]


_ensure_drjit_llvm_path()

from . import io
from . import optical_depth
from . import domain
from .optical_depth import vertically_integrated_optical_depth
from .domain import DomainGeometry, compute_domain_geometry
from .field import CloudField, load
from .camera import Camera


def glimpse(*args, **kwargs):
    """Lazily call :func:`cloudyview.glimpse.glimpse`."""
    return import_module(".glimpse", __name__).glimpse(*args, **kwargs)


def witness(*args, **kwargs):
    """Lazily call :func:`cloudyview.witness.witness`."""
    return import_module(".witness", __name__).witness(*args, **kwargs)


def behold(*args, **kwargs):
    """Lazily call :func:`cloudyview.behold.behold`."""
    return import_module(".behold", __name__).behold(*args, **kwargs)


def save_image(*args, **kwargs):
    """Lazily call :func:`cloudyview.image.save_image`."""
    return import_module(".image", __name__).save_image(*args, **kwargs)


def quantize_uint8(*args, **kwargs):
    """Lazily call :func:`cloudyview.image.quantize_uint8`."""
    return import_module(".image", __name__).quantize_uint8(*args, **kwargs)


class _CloudyViewModule(ModuleType):
    """Keep the public render functions from being replaced by their modules.

    ``cv.witness`` is the render function, and ``cloudyview.witness`` is also
    a submodule. Python's import machinery sets the package attribute to the
    submodule when it is loaded, which would replace the function; this keeps
    the function.
    """

    _public_renderers = frozenset(("glimpse", "witness", "behold"))

    def __setattr__(self, name, value):
        if name in self._public_renderers and isinstance(value, ModuleType):
            self.__dict__[f"_{name}_module"] = value
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CloudyViewModule

__all__ = ["io", "optical_depth", "domain",
           "vertically_integrated_optical_depth",
           "DomainGeometry", "compute_domain_geometry",
           "CloudField", "load", "Camera",
           "glimpse", "witness", "behold", "save_image", "quantize_uint8"]
