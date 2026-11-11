"""A soar still's PNG metadata can be read back in Python."""

import json
from pathlib import Path

import numpy as np

from cloudyview.camera import Camera
from cloudyview.image import PNG_TEXT_KEY, camera_from_metadata, read_metadata


def test_a_stills_metadata_reads_back(tmp_path: Path):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    # What web/soar/capture.js embeds, written here the way it writes it: one
    # tEXt chunk of JSON under PNG_TEXT_KEY.
    written = {
        "schema": "cloudyview.render.v1",
        "source": {"path": "/data/QC.nc", "ice_path": "/data/QI.nc"},
        "camera": {"position": [0.25, -0.5, -0.75], "azimuth": 123.5,
                   "elevation": 12.25, "fov": 92.0},
        "sun": {"azimuth": 20.0, "elevation": 55.0},
        "reproduction_command": "witness /data/QC.nc --quality high",
    }
    info = PngInfo()
    info.add_text(PNG_TEXT_KEY, json.dumps(written))
    png = tmp_path / "still.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(png, pnginfo=info)

    read_back = read_metadata(png)
    assert read_back == written
    assert camera_from_metadata(read_back) == Camera(
        position=(0.25, -0.5, -0.75), azimuth=123.5, elevation=12.25, fov=92.0)


def test_a_png_without_metadata_says_so(tmp_path: Path):
    from PIL import Image
    import pytest

    png = tmp_path / "plain.png"
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(png)
    with pytest.raises(KeyError, match="render_metadata"):
        read_metadata(png)
