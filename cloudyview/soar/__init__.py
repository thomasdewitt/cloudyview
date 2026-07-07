"""cloudyview.soar: wgpu/WGSL real-time volume renderer (spike).

Proof-of-pipeline for the interactive fly-through engine described in
docs/architecture.md. The extinction volume lives resident on the GPU as a
3D texture; frames are raymarched by `raymarch.wgsl`.

Requires the ``interactive`` extra (wgpu, glfw):

    uv sync --extra interactive

Offscreen:

    import cloudyview as cv
    from cloudyview.soar import InteractiveRenderer

    field = cv.load("cloud.nc")
    r = InteractiveRenderer(field)
    img = r.render(cv.Camera(), size=(960, 540))   # (H, W, 3) uint8

Windowed fly-through:

    uv run python -m cloudyview.soar cloud.nc [--ice ice.nc]

ESC opens the control-center menu (resume, open file, render in behold,
fullscreen, quit). F12 writes a metadata-bearing PNG screenshot.
"""

from .engine import InteractiveRenderer, request_device

__all__ = ["InteractiveRenderer", "request_device"]
