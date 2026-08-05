"""The camera conventions the three renderers have to share.

A camera handed from soar to witness to behold is the same camera or the
hand-off is worthless. Two things had drifted (fixed 2026-08-05): `fov` was
read as horizontal by behold and vertical by the other two, and relative
height meant "above the surface" everywhere except behold, where it meant
"up from the bottom of whatever slab was loaded".
"""

import importlib

import numpy as np
import pytest

from cloudyview.camera import Camera
from cloudyview.domain import compute_domain_geometry

behold = importlib.import_module("cloudyview.behold")


def _geometry(z_bottom: float, z_top: float, nz: int = 64):
    """Geometry for a domain whose data occupies z_bottom..z_top."""
    dz = (z_top - z_bottom) / nz
    z = z_bottom + dz * (np.arange(nz) + 0.5)
    x = y = 100.0 * (np.arange(32) + 0.5)
    return compute_domain_geometry(x, y, z, 32, 32, nz)


def test_domain_geometry_reports_absolute_cell_edges():
    geom = _geometry(1000.0, 5000.0)

    assert geom.z_bottom == pytest.approx(1000.0)
    assert geom.z_top == pytest.approx(5000.0)
    assert geom.height_z == pytest.approx(geom.z_top - geom.z_bottom)


def test_ground_level_domain_leaves_relative_height_untouched():
    """The old mapping, for the only case in which it was ever right."""
    geom = _geometry(0.0, 5000.0)

    for rel_z in (-0.999, -0.5, 0.0, 0.87):
        assert behold._relative_z_to_cube(rel_z, geom) == pytest.approx(rel_z)


def test_elevated_domain_keeps_the_camera_at_its_real_altitude():
    """A nest based at 1 km must not slam the surface up to its floor.

    rel_z = -0.99 is 25 m above the sea in witness and soar. Read against
    the nest's own slab it used to come out at ~1 km — inside the volume,
    a kilometre above where the view was framed.
    """
    geom = _geometry(1000.0, 5000.0)
    rel_z = -0.99

    cube_z = behold._relative_z_to_cube(rel_z, geom)

    # Back to metres: the cube spans the data slab, [-1, 1] over 4000 m.
    metres = geom.z_bottom + (cube_z + 1.0) * 0.5 * geom.height_z
    assert metres == pytest.approx(0.005 * geom.z_top)   # 25 m
    assert cube_z < -1.0                                 # below the volume


def _wedge_half_angle_deg(fov: float, render_aspect: float) -> float:
    """Half-angle of glimpse's top-down camera wedge, in degrees."""
    from cloudyview.glimpse import _build_camera_overlay

    overlay = _build_camera_overlay(
        image_shape=(128, 128),
        camera_position=[0.0, 0.0, -0.9],
        camera_azimuth=0.0,
        camera_elevation=0.0,
        camera_fov=fov,
        render_aspect=render_aspect,
    )
    (lx, ly), (rx, ry) = overlay["fov_endpoints"]
    cx, cy = overlay["camera_xy"]
    left = np.array([lx - cx, ly - cy])
    right = np.array([rx - cx, ry - cy])
    cos = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return 0.5 * float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


@pytest.mark.parametrize("render_aspect", [1.0, 1.5, 0.75])
def test_fov_spans_the_width_whatever_the_aspect(render_aspect):
    """fov is horizontal, so the wedge is fov wide at any image shape.

    Under the old vertical reading this wedge widened with the aspect
    ratio while behold's frame stayed put — the same camera, two views.
    """
    assert _wedge_half_angle_deg(60.0, render_aspect) == pytest.approx(30.0,
                                                                      abs=1e-6)
