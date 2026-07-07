"""PNG metadata helpers for CloudyView render outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .camera import Camera

PNG_TEXT_KEY = "cloudyview.render_metadata"
SCHEMA = "cloudyview.render.v1"


def _jsonable(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-round-tripped dict with non-JSON scalars stringified."""
    return json.loads(json.dumps(dict(value), default=str))


def embed_metadata(
    png_path_or_image,
    metadata: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
):
    """Embed CloudyView metadata in PNG tEXt chunks.

    If ``png_path_or_image`` is a path, the PNG is rewritten in place unless
    ``output_path`` is supplied. If it is a PIL image, ``output_path`` is
    required and the image is written there.
    """
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    clean = _jsonable(metadata)
    pnginfo = PngInfo()
    pnginfo.add_text(PNG_TEXT_KEY, json.dumps(clean, sort_keys=True))

    if isinstance(png_path_or_image, (str, Path)):
        src_path = Path(png_path_or_image)
        dst_path = Path(output_path) if output_path is not None else src_path
        with Image.open(src_path) as image:
            image.load()
            image.save(dst_path, format="PNG", pnginfo=pnginfo)
        return dst_path

    if not isinstance(png_path_or_image, Image.Image):
        raise TypeError(
            "embed_metadata expects a PNG path or PIL.Image.Image; "
            f"got {type(png_path_or_image)!r}."
        )
    if output_path is None:
        raise ValueError("output_path is required when embedding into a PIL image.")
    dst_path = Path(output_path)
    png_path_or_image.save(dst_path, format="PNG", pnginfo=pnginfo)
    return dst_path


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read CloudyView metadata from a PNG written by ``embed_metadata``."""
    from PIL import Image

    with Image.open(path) as image:
        raw = image.info.get(PNG_TEXT_KEY)
    if raw is None:
        raise KeyError(f"{path!s} has no {PNG_TEXT_KEY!r} PNG metadata chunk.")
    return json.loads(raw)


def reconstruct_camera_and_paths(
    metadata: Mapping[str, Any],
) -> tuple[Camera, dict[str, str | None]]:
    """Reconstruct the ``Camera`` and source paths stored in render metadata."""
    camera_meta = metadata["camera"]
    source_meta = metadata.get("source", {})
    camera = Camera(
        position=tuple(camera_meta["position"]),
        azimuth=camera_meta["azimuth"],
        elevation=camera_meta["elevation"],
        fov=camera_meta["fov"],
    )
    return camera, {
        "source": source_meta.get("path"),
        "ice": source_meta.get("ice_path"),
    }


def build_render_metadata(
    field,
    camera: Camera,
    *,
    sun_azimuth: float,
    sun_elevation: float,
    renderer: str,
    quality: str | None = None,
    reproduction_command: str,
    timestamp: datetime | None = None,
    render_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard metadata dict for soar screenshots and behold PNGs."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    render: dict[str, Any] = {"renderer": renderer}
    if quality is not None:
        render["quality"] = quality
    if render_options:
        render.update(dict(render_options))
    return {
        "schema": SCHEMA,
        "source": {
            "path": getattr(field, "source", None),
            "ice_path": getattr(field, "ice_source", None),
            "liquid_var": getattr(field, "liquid_var", None),
            "ice_var": getattr(field, "ice_var", None),
        },
        "camera": {
            "position": [float(v) for v in camera.position],
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
            "fov": float(camera.fov),
        },
        "sun": {
            "azimuth": float(sun_azimuth),
            "elevation": float(sun_elevation),
        },
        "render": render,
        "cloudyview_version": __version__,
        "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
        "reproduction_command": reproduction_command,
    }
