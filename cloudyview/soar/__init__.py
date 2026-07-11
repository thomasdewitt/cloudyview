"""cloudyview.soar: wgpu/WGSL real-time volume renderer (spike).

Proof-of-pipeline for the interactive fly-through engine described in
docs/architecture.md. The extinction volume lives resident on the GPU as a
3D texture; frames are raymarched by `raymarch.wgsl`.

Requires the ``interactive`` extra (wgpu, glfw, imgui-bundle):

    uv sync --extra interactive

Offscreen:

    import cloudyview as cv
    from cloudyview.soar import InteractiveRenderer

    field = cv.load("cloud.nc")
    r = InteractiveRenderer(field)
    img = r.render(cv.Camera(), size=(960, 540))   # (H, W, 3) uint8

Windowed fly-through:

    uv run python -m cloudyview.soar cloud.nc [--ice ice.nc]

ESC opens the in-window pause menu; ESC resumes from the top level and backs
out of submenus. F12 writes a metadata-bearing PNG screenshot.
"""

from .engine import (
    QUALITY_PRESETS,
    InteractiveRenderer,
    QualityPreset,
    render_target_size,
    request_device,
)

__all__ = [
    "InteractiveRenderer",
    "QualityPreset",
    "QUALITY_PRESETS",
    "render_target_size",
    "request_device",
]
