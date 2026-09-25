"""A flora decal averages, seen from above, to what the field says.

raymarch.wgsl df_disc_cell covers a ray's chord through a plant's crown
envelope with 1 - exp(-sigma chord), sigma set from DF_DECAL_TAU0_*, the
optical depth vertically through the crown's centre. Over the crown's disc
from above, the chord is tau0 sqrt(1 - r^2) deep, and the mean cover is

    1 - 2 (1 - e^-tau0 (1 + tau0)) / tau0^2,

which must equal the field's opacity for the species (DF_FAR_*_KAPPA; a
barrel 0.95, a heap 0.5), or the decal and the field it fades into would
disagree where one hands over to the other. This checks the closed form
against the shader's constants, and the closed form against a direct
quadrature, so a change to either kappa or tau0 that forgets the other
fails here.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

SHADER = (Path(__file__).resolve().parents[1]
          / "cloudyview" / "soar" / "raymarch.wgsl").read_text()


def const(name):
    m = re.search(rf"^const {name}: f32 = ([0-9.]+);", SHADER, re.M)
    assert m, name
    return float(m.group(1))


def disc_mean(tau0):
    return 1.0 - 2.0 * (1.0 - math.exp(-tau0) * (1.0 + tau0)) / tau0 ** 2


CASES = [("DF_DECAL_TAU0_CREO", const("DF_FAR_CREO_KAPPA")),
         ("DF_DECAL_TAU0_BURS", const("DF_FAR_BURS_KAPPA")),
         ("DF_DECAL_TAU0_PEAR", const("DF_FAR_PEAR_KAPPA")),
         ("DF_DECAL_TAU0_PV", const("DF_FAR_PV_KAPPA")),
         ("DF_DECAL_TAU0_BARREL", 0.95),
         ("DF_DECAL_TAU0_ROCK", 0.5)]


@pytest.mark.parametrize("tau_name,kappa", CASES)
def test_decal_mean_is_the_fields_opacity(tau_name, kappa):
    assert disc_mean(const(tau_name)) == pytest.approx(kappa, abs=2e-4)


@pytest.mark.parametrize("tau0", [0.5, 1.3, 3.4, 6.3])
def test_closed_form_matches_quadrature(tau0):
    r = np.linspace(0.0, 1.0, 400001)
    cover = 1.0 - np.exp(-tau0 * np.sqrt(1.0 - r * r))
    assert np.trapezoid(cover * 2.0 * r, r) == pytest.approx(
        disc_mean(tau0), abs=1e-6)
