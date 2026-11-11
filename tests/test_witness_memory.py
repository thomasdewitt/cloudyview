"""The Python host builds and uploads a field without whole-field copies.

Rendering the fine utah-convection demo (2048 x 2048 x 495, liquid and ice)
from Python ran out of memory past 32 GB (2026-11-09): witness._field_level
built the whole field's extinction (five field-sized float32 arrays alive at
the liquid + ice sum), copied it to float64, then copied the occupied band;
the upload made a whole-field fp16 copy and a tobytes() twin of it. Now the
extinction is built in x-slabs, twice (the planes' peaks, then the band), and
uploaded in x-slabs: measured on that field, the heap peak over the field
fell to 1.05 x one float32 field array (8.3 GB; from about 5 x, 41.5 GB, by
the coarse field's measured 5.00 x), and the uploaded texture and the image
hash the same as before.

These hold the result to the old whole-field formula, bit for bit, and the
memory to a band's worth; no GPU.
"""

import importlib
import tracemalloc

import numpy as np
import pytest

from cloudyview import optical_depth
from cloudyview.soar.renderer import SoarRenderer
from cloudyview.field import CloudField

# The module, not the function the package re-exports under the same name.
witness = importlib.import_module("cloudyview.witness")


def _field(nx=24, ny=20, nz=40, ice=True, seed=3, z_dtype=np.float32):
    rng = np.random.default_rng(seed)
    lwc = rng.gamma(0.4, 0.3, (nx, ny, nz)).astype(np.float32)
    iwc = rng.gamma(0.3, 0.2, (nx, ny, nz)).astype(np.float32) if ice else None
    # Empty sky below and above, so the band is a crop at both ends.
    for a in (lwc, iwc) if ice else (lwc,):
        a[:, :, :7] = 0.0
        a[:, :, -9:] = 0.0
    z = (500.0 + 37.5 * np.arange(nz)).astype(z_dtype)
    x = 50.0 * np.arange(nx) + 25.0
    y = 50.0 * np.arange(ny) + 25.0
    return CloudField(lwc=lwc, iwc=iwc, x=x, y=y, z=z)


def _old_field_level_sigma(field, multiplier):
    """The whole-field construction this replaced, verbatim in effect."""
    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < witness.ICE_NEGLIGIBLE_G_KG:
        iwc = None
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=witness.RE_LIQUID_UM, iwc=iwc,
        re_ice=witness.RE_ICE_UM)
    sigma = np.ascontiguousarray(sigma * multiplier, dtype=np.float64)
    sigma, z_crop, (lo, hi) = witness.crop_empty_z(sigma, field.z)
    return np.ascontiguousarray(sigma)


@pytest.fixture
def multiplier(monkeypatch):
    m = 1.7
    monkeypatch.setattr(witness, "EXTINCTION_MULTIPLIER", m)
    return m


@pytest.mark.parametrize("ice", [True, False])
@pytest.mark.parametrize("z_dtype", [np.float32, np.float64])
def test_the_band_is_the_old_whole_field_result_bit_for_bit(
        monkeypatch, multiplier, ice, z_dtype):
    # Slabs of 2-3 x-rows, so the tiling is exercised many times over.
    monkeypatch.setattr(witness, "_SLAB_VOXELS", 1000)
    field = _field(ice=ice, z_dtype=z_dtype)
    level = witness._field_level(field, "single")
    old = _old_field_level_sigma(field, multiplier)
    assert level.sigma.shape == old.shape
    assert old.shape[2] < field.lwc.shape[2]          # it did crop
    # The values themselves: float64 holds the float32 ones exactly.
    assert np.array_equal(level.sigma.astype(np.float64), old)
    # And what the texture gets.
    assert np.array_equal(level.sigma.astype(np.float16).view(np.uint16),
                          old.astype(np.float16).view(np.uint16))


def test_negligible_ice_is_still_dropped(monkeypatch, multiplier):
    monkeypatch.setattr(witness, "_SLAB_VOXELS", 1000)
    field = _field()
    field.iwc[...] = witness.ICE_NEGLIGIBLE_G_KG * 0.5
    level = witness._field_level(field, "single")
    assert np.array_equal(level.sigma.astype(np.float64),
                          _old_field_level_sigma(field, multiplier))


def test_the_peak_is_a_band_not_five_fields(monkeypatch):
    monkeypatch.setattr(witness, "_SLAB_VOXELS", 4000)
    field = _field(nx=96, ny=64, nz=60)
    tracemalloc.start()
    try:
        level = witness._field_level(field, "single")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    one = field.lwc.nbytes
    # The band (float32) plus a slab's temporaries; the old construction
    # peaked at five float32 fields, then held a float64 one beside a
    # float32 one.
    assert level.sigma.dtype == np.float32
    assert peak < level.sigma.nbytes + 0.25 * one, (peak, one)


class _Queue:
    def __init__(self):
        self.writes = []

    def write_texture(self, destination, data, layout, size):
        self.writes.append((destination["origin"], bytes(memoryview(data)),
                            layout, size))


def test_the_upload_in_slabs_writes_what_one_write_did(monkeypatch):
    r = object.__new__(SoarRenderer)
    r.device = type("Device", (), {"queue": _Queue()})()
    monkeypatch.setattr(SoarRenderer, "_UPLOAD_SLAB_TEXELS", 7 * 11 * 3)
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        r.device.queue.writes.clear()
        sigma = (rng.random((10, 11, 7)) * 1e-2).astype(dtype)
        r._write_r16_volume("tex", sigma)
        writes = r.device.queue.writes
        assert len(writes) == 4                          # 3 + 3 + 3 + 1 rows
        x = 0
        for origin, data, layout, size in writes:
            assert origin == (0, 0, x)
            assert layout == {"bytes_per_row": 7 * 2, "rows_per_image": 11}
            assert size[:2] == (7, 11)
            x += size[2]
        assert x == 10
        whole = np.ascontiguousarray(sigma, dtype=np.float16).tobytes()
        assert b"".join(w[1] for w in writes) == whole


def test_clamping_keeps_its_values_and_leaves_the_callers_array_alone():
    rng = np.random.default_rng(1)
    given = rng.normal(0.0, 1.0, (8, 9, 10)).astype(np.float32)
    given[0, 0, 0] = -0.0
    before = given.copy()
    field = CloudField(lwc=given, x=np.arange(8.0), y=np.arange(9.0),
                       z=np.arange(10.0))
    expected = np.where(before < 0, 0.0, before).astype(np.float32)
    assert np.array_equal(field.lwc.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(given.view(np.uint32), before.view(np.uint32))
    # A float64 input is converted, and that conversion is the only copy.
    f64 = before.astype(np.float64)
    field = CloudField(lwc=f64, x=np.arange(8.0), y=np.arange(9.0),
                       z=np.arange(10.0))
    assert np.array_equal(field.lwc.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(f64, before.astype(np.float64))
