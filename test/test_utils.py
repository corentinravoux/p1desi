"""Unit tests for :mod:`p1desi.utils`."""

import numpy as np
import pytest

from p1desi import utils


def test_return_key():
    assert utils.return_key({"a": 1}, "a", 99) == 1
    assert utils.return_key({"a": 1}, "b", 99) == 99


def test_conversion_factor_positive():
    f = utils.return_conversion_factor(2.2)
    assert f > 0
    # factor is c / (1+z) / lambda_Lya
    expected = utils.speed_light / (1 + 2.2) / utils.lambdaLy
    assert f == pytest.approx(expected)


def test_k_unit_roundtrip():
    k_aa = np.array([0.01, 0.05, 0.1])
    for z in (2.2, 3.0, 3.8):
        k_kms = utils.kAAtokskm(k_aa, z=z)
        back = utils.kskmtokAA(k_kms, z=z)
        np.testing.assert_allclose(back, k_aa, rtol=1e-12)


def test_k_unit_direction():
    # s/km wavenumbers are numerically smaller than Angstrom^-1 ones
    assert utils.kAAtokskm(0.1, z=2.2) < 0.1


def test_label_formatters():
    # formatters return strings with three decimals matching the conversion
    s1 = utils.kAAtokskm_label(0.1, z=2.2)
    s2 = utils.kskmtokAA_label(0.001, z=2.2)
    assert isinstance(s1, str) and isinstance(s2, str)
    assert len(s1.split(".")[-1]) == 3
    assert float(s1) == pytest.approx(round(utils.kAAtokskm(0.1, z=2.2), 3))
    assert float(s2) == pytest.approx(round(utils.kskmtokAA(0.001, z=2.2), 3))


def test_doublet_constants():
    # Si IV / C IV velocity separations are positive and ordered sensibly
    assert utils.vsi4 > 0
    assert utils.vc4 > 0
    assert utils.lsi4 == pytest.approx(utils.si4_2 - utils.si4_1)
