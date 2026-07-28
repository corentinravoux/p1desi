"""Unit tests for :mod:`p1desi.resolution`."""

import numpy as np
import pytest

from p1desi import resolution


def test_model_resolution_shape_and_limit():
    k = np.linspace(0.001, 0.1, 50)
    r = resolution.model_resolution(k, dl=0.8)
    assert r.shape == k.shape
    # correction decreases with scale (larger k -> smaller correction)
    assert r[0] > r[-1]
    # near k -> 0 the resolution window tends to 1
    assert resolution.model_resolution(np.array([1e-6]), dl=0.8)[0] == \
        pytest.approx(1.0, abs=1e-6)


def test_fit_model_resolution_recovers_dl():
    k = np.linspace(0.001, 0.1, 100)
    dl_true = 0.7
    y = resolution.model_resolution(k, dl_true)
    dy = np.full_like(y, 0.001)
    popt = resolution.fit_model_resolution(k, y, dy)
    assert popt[0] == np.float64(popt[0])
    np.testing.assert_allclose(abs(popt[0]), dl_true, rtol=1e-3)


def test_fit_model_resolution_ignores_nans():
    k = np.linspace(0.001, 0.1, 100)
    y = resolution.model_resolution(k, 0.7)
    dy = np.full_like(y, 0.001)
    y[10] = np.nan
    k[20] = np.nan
    popt = resolution.fit_model_resolution(k, y, dy)  # must not raise
    np.testing.assert_allclose(abs(popt[0]), 0.7, rtol=1e-2)
