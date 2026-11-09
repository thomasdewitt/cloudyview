"""Every pixel of a near-ground view is finite, on every surface.

A NaN in one pass of soar's accumulation survives into the average and, in
the browser, sticks for the whole hold. The city had one (2026-09-25): an
advert screen's product backdrop took pow() of a negative number just above
its panel, where the result is masked to zero — but NaN * 0 is NaN, so a
far screen, reached only by a few jittered rays near the horizon, left a
handful of NaN pixels in every held frame. This renders views from just
above the ground, where the far field is widest, at High (32 jittered
passes, which is what found it) in linear mode, and fails on any non-finite
pixel.
"""

import numpy as np
import pytest

try:
    from .conftest import SOAR_DATA_FILE, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, soar_gpu_adapter

# (position in the app's relative coordinates, azimuth, elevation, fov):
# near the floor of the domain box, looking just below the horizon, so a
# view holds the street, the skyline and the sky above it.
VIEWS = [
    ((0.0, 0.0, -0.95), 30.0, -6.0, 70.0),
    ((0.0, 0.0, -0.97), 30.0, -6.0, 70.0),
    ((0.3, -0.4, -0.99), 200.0, -2.0, 70.0),
]
# Soar's cyberpunk mode flies this district (witness's own example).
CITY_OFFSET = (8100.0, -44190.0)
SIZE = (320, 180)


@pytest.fixture(scope="module")
def field():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    import cloudyview as cv
    return cv.load(str(SOAR_DATA_FILE))


@pytest.mark.parametrize("surface", ["city", "ocean"])
@pytest.mark.parametrize("view", VIEWS, ids=["z95", "z97", "z99"])
def test_near_ground_pixels_are_finite(field, surface, view):
    import cloudyview as cv
    from cloudyview import Camera
    position, azimuth, elevation, fov = view
    img = cv.witness(
        field, Camera(position=position, azimuth=azimuth,
                      elevation=elevation, fov=fov),
        size=SIZE, periodic=True, quality="high", surface=surface,
        city_offset=CITY_OFFSET if surface == "city" else None,
        return_linear=True, verbose=False)
    bad = ~np.isfinite(img).all(axis=-1)
    ys, xs = np.nonzero(bad)
    assert not bad.any(), (
        f"{int(bad.sum())} non-finite pixel(s) on the {surface}, first at "
        f"(row, col) {list(zip(ys[:5].tolist(), xs[:5].tolist()))}")
