"""Render the TWPICE deep-convection regression view.

Usage: uv run python temp/tuning-2026-08-11/render_twpice.py <out.png> [--set k=v ...]
"""
import sys
import numpy as np
from PIL import Image

import cloudyview as cv
from cloudyview.camera import Camera

FIELD = "data/TWPICE_subvolume_256x256_5km.nc"

# Mid-distance view of the convective mass, sun matching the tuning scene.
CAMERA = Camera(position=(-0.85, -0.85, -0.95), azimuth=45.0, elevation=18.0,
                fov=80.0)


def main():
    out = sys.argv[1]
    field = cv.load(FIELD)
    img = cv.witness(field, CAMERA, size=(1600, 900),
                     sun_azimuth=20.0, sun_elevation=55.0,
                     tone_map_gamma=1.66, accumulate=64)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
