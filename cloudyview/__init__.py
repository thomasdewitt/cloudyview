"""CloudyView: 3D cloud field visualization toolkit."""

__version__ = "0.1.0"

from . import io
from . import basic_render
from . import optical_depth

# Lazy import for radiative_transfer (requires mitsuba)
def __getattr__(name):
    if name == "radiative_transfer":
        from . import radiative_transfer
        return radiative_transfer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["io", "basic_render", "optical_depth", "radiative_transfer"]
