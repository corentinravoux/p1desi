"""Unit tests for :mod:`p1desi.pk_io` helpers, Pk methods and external readers."""

import numpy as np
import pytest

from p1desi import pk_io


# --- module-level covariance / variance helpers ----------------------------


def test_posify_variance_interpolates_negatives():
    var = np.array([1.0, 2.0, -5.0, 4.0, 5.0])
    out = pk_io.posify_variance(var)
    assert np.all(out > 0)
    # the interpolated point sits between its positive neighbours
    assert 2.0 < out[2] < 4.0


def test_posify_variance_all_nan_passthrough():
    var = np.full(4, np.nan)
    out = pk_io.posify_variance(var)
    assert np.all(np.isnan(out))


def test_smooth_variance_preserves_nans_and_length():
    var = np.abs(np.sin(np.linspace(0, 3, 60))) + 0.1
    var[5] = np.nan
    out = pk_io.smooth_variance(var, smoothing_window=11, smoothing_polynomial=3)
    assert out.shape == var.shape
    assert np.isnan(out[5])
    assert np.all(np.isfinite(out[~np.isnan(out)]))


def test_regularize_covariance_positive_definite():
    # a non-positive-definite symmetric matrix
    m = np.array([[1.0, 2.0], [2.0, 1.0]])
    reg = pk_io.regularize_covariance(m, eigval_min=1e-8)
    eigvals = np.linalg.eigvals(reg)
    assert np.all(eigvals.real > 0)


def test_regularize_covariance_nonfinite_passthrough():
    m = np.array([[np.nan, 0.0], [0.0, 1.0]])
    out = pk_io.regularize_covariance(m)
    assert np.isnan(out[0, 0])


# --- Pk covariance methods (using the synthetic fixture) -------------------


def test_use_covariance_as_error(pk):
    pk.use_covariance_as_error(use_boot=True)
    for z in pk.zbin:
        # our synthetic covariance diagonal is err**2, so err is recovered
        np.testing.assert_allclose(pk.err[z], np.full(len(pk.k[z]), 1.0) * pk.err[z])
        np.testing.assert_allclose(pk.norm_err[z], pk.k[z] * pk.err[z] / np.pi)


def test_multiplicative_covariance_correction(pk):
    z = pk.zbin[0]
    before = pk.cov[z].copy()
    pk.multiplicative_covariance_correction(correction_term=2.0)
    np.testing.assert_allclose(pk.cov[z], before * 2.0)


def test_posify_covariance_diagonal(pk):
    z = pk.zbin[0]
    nk = len(pk.k[z])
    cov = pk.cov[z].reshape(nk, nk)
    cov[3, 3] = -1.0
    pk.cov[z] = np.ravel(cov)
    pk.posify_covariance_diagonal()
    diag = pk.cov[z].reshape(nk, nk).diagonal()
    assert np.all(diag > 0)


def test_regularize_and_smooth_diagonal_run(pk):
    # these must run without error and keep the covariance shape
    pk.regularize_covariance(eigval_min=1e-12)
    pk.smooth_covariance_diagonal(smoothing_window=5, smoothing_polynomial=2)
    for z in pk.zbin:
        assert pk.cov[z].size == len(pk.k[z]) ** 2


def test_correct_noise_global_reduces_power(pk):
    z = pk.zbin[0]
    p_before = pk.p[z].copy()
    pk.correct_noise_global(1e-3)
    assert np.all(pk.p[z] < p_before)


def test_compute_additional_stats(pk):
    pk.compute_additional_stats()
    for z in pk.zbin:
        assert z in pk.err_noiseoverraw
        assert np.all(np.isfinite(pk.err_noiseoverraw[z]))


# --- redshift / wavenumber reductions --------------------------------------


def test_meanpkz(pk):
    mz = pk_io.MeanPkZ.init_from_pk(pk)
    assert len(mz.k) == len(pk.k[pk.zbin[0]])
    # the averaged power lies within the per-bin range
    stack = np.array([pk.p[z] for z in pk.zbin])
    assert np.all(mz.p <= stack.max(axis=0) + 1e-12)
    assert np.all(mz.p >= stack.min(axis=0) - 1e-12)


def test_meanpkk(pk):
    mk = pk_io.MeanPkK.init_from_pk(pk)
    for z in pk.zbin:
        assert np.isfinite(mk.p[z])
        assert mk.p[z] == pytest.approx(np.mean(pk.p[z]))


# --- external-format readers ----------------------------------------------


def test_pkeboss_reader(tmp_path):
    # columns: z, k, p, err_stat, p_noise, (unused), err_syst
    rows = []
    for z in (2.2, 2.4):
        for k in (0.01, 0.02, 0.03):
            rows.append([z, k, 0.1, 0.01, 0.02, 0.0, 0.005])
    path = tmp_path / "eboss.txt"
    np.savetxt(str(path), np.array(rows))
    pk = pk_io.PkEboss.read_from_file(str(path))
    assert pk.velunits is True
    np.testing.assert_array_equal(pk.zbin, [2.2, 2.4])
    z = pk.zbin[0]
    # total error is stat and syst added in quadrature
    np.testing.assert_allclose(pk.err[z], np.sqrt(0.01 ** 2 + 0.005 ** 2))


def test_pkhr_reader(tmp_path):
    # pipe-delimited, one header row, columns index|z|k|p|err
    lines = ["idx|z|k|p|err"]
    for z in (2.2, 2.4):
        for k in (0.01, 0.02):
            lines.append(f"0|{z}|{k}|0.1|0.01")
    path = tmp_path / "hr.txt"
    path.write_text("\n".join(lines) + "\n")
    pk = pk_io.PkHR.read_from_file(str(path))
    assert pk.velunits is True
    np.testing.assert_array_equal(pk.zbin, [2.2, 2.4])
    z = pk.zbin[0]
    np.testing.assert_allclose(pk.norm_p[z], pk.k[z] * pk.p[z] / np.pi)
