"""End-to-end tests: synthetic picca FITS -> Pk -> reductions / cleaning.

These exercise the central I/O path (:meth:`p1desi.pk_io.Pk.read_from_picca`)
against a synthetic but format-faithful picca mean-P1D file, then run the
downstream reductions and the covariance-cleaning pipeline.
"""

import numpy as np
import pytest

from p1desi import pk_io

from conftest import make_picca_pk_fits


def test_read_from_picca_roundtrip(tmp_path):
    zbins, k = make_picca_pk_fits(tmp_path / "m.fits", zbins=(2.2, 2.4, 2.6), nk=12)
    pk = pk_io.Pk.read_from_picca(str(tmp_path / "m.fits"))

    np.testing.assert_array_equal(pk.zbin, zbins)
    assert pk.velunits is True
    assert pk.number_qso == 1000
    for z in pk.zbin:
        assert len(pk.k[z]) == 12
        np.testing.assert_allclose(pk.k[z], k)
        assert pk.number_chunks[z] == 100
        # normalised power is k*P/pi
        np.testing.assert_allclose(pk.norm_p[z], pk.k[z] * pk.p[z] / np.pi, rtol=1e-6)
        # covariance HDU was read
        assert pk.cov[z].size == 12 * 12


def test_read_from_picca_without_covariance(tmp_path):
    make_picca_pk_fits(tmp_path / "m.fits", with_cov=True)
    make_picca_pk_fits(tmp_path / "nocov.fits", with_cov=False)
    pk = pk_io.Pk.read_from_picca(str(tmp_path / "nocov.fits"))
    # no covariance HDU -> empty covariance dicts, but the measurement still loads
    assert pk.cov == {}
    for z in pk.zbin:
        assert np.all(np.isfinite(pk.p[z]))


def test_angstrom_units_file(tmp_path):
    make_picca_pk_fits(tmp_path / "aa.fits", velunits=False)
    pk = pk_io.Pk.read_from_picca(str(tmp_path / "aa.fits"))
    assert pk.velunits is False


def test_covariance_cleaning_pipeline(tmp_path):
    pytest.importorskip("sgolay2")  # 2D smoothing needs sgolay2
    make_picca_pk_fits(tmp_path / "m.fits", nk=40)
    pk = pk_io.Pk.read_from_picca(str(tmp_path / "m.fits"))
    pk.correct_covariance()
    pk.use_covariance_as_error()
    for z in pk.zbin:
        cov = pk.cov[z].reshape(len(pk.k[z]), len(pk.k[z]))
        # cleaned covariance is finite and positive-definite
        assert np.all(np.isfinite(cov))
        assert np.all(np.linalg.eigvals(cov).real > 0)
        assert np.all(pk.err[z] > 0)


def test_covariance_cleaning_steps_without_sgolay(tmp_path):
    # the steps that do not need sgolay2 must run and keep a usable covariance
    make_picca_pk_fits(tmp_path / "m.fits", nk=30)
    pk = pk_io.Pk.read_from_picca(str(tmp_path / "m.fits"))
    pk.posify_covariance_diagonal()
    pk.smooth_covariance_diagonal(smoothing_window=11, smoothing_polynomial=3)
    pk.multiplicative_covariance_correction(0.9)
    pk.regularize_covariance()
    pk.use_covariance_as_error()
    for z in pk.zbin:
        assert np.all(pk.err[z] > 0)
