"""Unit tests for :mod:`p1desi.mocks` (chi-squared diagnostics)."""

import numpy as np
import pytest

from p1desi import mocks


def test_getchi2pdf_normalised():
    x, pdf = mocks.getChi2Pdf(10)
    assert len(x) == len(pdf) == 100
    assert np.all(pdf >= 0)
    # the mode sits near dof - 2 for a chi-squared distribution
    assert 6 < x[np.argmax(pdf)] < 10


def test_compute_chi2_self_is_zero_diag(pk):
    # a measurement compared to itself has zero chi-squared
    chi2, ndof, zbin = mocks.compute_chi2(
        [pk], pk, kmin=0.0, kmax=1.0, zmin=2.0, zmax=3.0, use_only_diag=True
    )
    assert chi2[0] == pytest.approx(0.0, abs=1e-9)
    assert ndof[0] > 0


def test_compute_chi2_self_is_zero_full_cov(pk):
    chi2, ndof, zbin = mocks.compute_chi2(
        [pk], pk, kmin=0.0, kmax=1.0, zmin=2.0, zmax=3.0, use_boot=True
    )
    assert chi2[0] == pytest.approx(0.0, abs=1e-9)


def test_compute_chi2_offset_positive(pk):
    import copy

    shifted = copy.deepcopy(pk)
    for z in shifted.zbin:
        shifted.p[z] = shifted.p[z] * 1.1  # 10% offset
    chi2, ndof, zbin = mocks.compute_chi2(
        [shifted], pk, kmin=0.0, kmax=1.0, zmin=2.0, zmax=3.0, use_only_diag=True
    )
    assert chi2[0] > 0


def test_compute_chi2_per_redshift(pk):
    chi2, ndof, zbin = mocks.compute_chi2(
        [pk], pk, zmin=2.0, zmax=3.0, use_only_diag=True, return_redshift_chi2=True
    )
    assert len(chi2) == len(zbin)
    for c in chi2:
        assert c[0] == pytest.approx(0.0, abs=1e-9)
