"""Tests for CloudyView PNG render metadata."""

from pathlib import Path

import numpy as np

from cloudyview.camera import Camera
from cloudyview.cloudfield import CloudField
from cloudyview.render_metadata import (
    build_render_metadata,
    embed_metadata,
    read_metadata,
    reconstruct_camera_and_paths,
)


def test_metadata_embed_read_reconstruct_camera_roundtrip(tmp_path: Path):
    from PIL import Image

    png_path = tmp_path / "render.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(png_path)

    field = CloudField(
        lwc=np.zeros((2, 2, 2), dtype=np.float32),
        x=np.arange(2),
        y=np.arange(2),
        z=np.arange(2),
        source="/data/QC.nc",
        ice_source="/data/QI.nc",
        liquid_var="QC",
        ice_var="QI",
    )
    camera = Camera(
        position=(0.25, -0.5, -0.75),
        azimuth=123.5,
        elevation=12.25,
        fov=92.0,
    )
    metadata = build_render_metadata(
        field,
        camera,
        sun_azimuth=20.0,
        sun_elevation=55.0,
        renderer="behold",
        quality="min",
        reproduction_command=(
            "behold /data/QC.nc min --gpu --camera-position 0.25 -0.5 -0.75"
        ),
    )

    embed_metadata(png_path, metadata)
    read_back = read_metadata(png_path)
    camera_back, paths = reconstruct_camera_and_paths(read_back)

    assert read_back["schema"] == "cloudyview.render.v1"
    assert read_back["source"]["path"] == "/data/QC.nc"
    assert read_back["source"]["ice_path"] == "/data/QI.nc"
    assert read_back["render"]["renderer"] == "behold"
    assert read_back["render"]["quality"] == "min"
    assert camera_back == camera
    assert paths == {"source": "/data/QC.nc", "ice": "/data/QI.nc"}
