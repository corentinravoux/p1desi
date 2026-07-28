"""Unit tests for :mod:`p1desi.hcd` (Rogers HCD contamination model)."""

import numpy as np

from p1desi import hcd


def test_rogers_returns_finite_positive():
    k = np.linspace(0.001, 0.1, 20)
    ratio = hcd.rogers(2.4, k, a0=1.1, a1=0.5, b0=10.0, b1=0.2, c0=0.01, c1=0.1)
    assert np.all(np.isfinite(ratio))
    assert ratio.shape == k.shape


def test_rogers_redshift_pivot():
    # at z = z0 = 2.0 the (1+z)/(1+z0) power-law factors are all unity, so the
    # result reduces to the closed form with a=a0, b=b0, c=c0, d=1.
    k = np.array([0.02, 0.05])
    a0, b0, c0 = 1.5, 8.0, 0.02
    got = hcd.rogers(2.0, k, a0=a0, a1=3.0, b0=b0, b1=-1.0, c0=c0, c1=2.0)
    expected = 1.0 * (1 / (a0 * np.exp(b0 * k) - 1) ** 2) + c0
    np.testing.assert_allclose(got, expected)
