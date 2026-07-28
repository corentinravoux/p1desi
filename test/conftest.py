"""Shared fixtures and helpers for the p1desi test suite.

p1desi ships no reference P1D FITS files, so the tests synthesise a valid
picca-format mean-P1D file (three HDUs: mean table, per-bin metadata and
covariance) with :func:`make_picca_pk_fits` and read it back through the public
:class:`p1desi.pk_io.Pk` API. Everything runs in a temporary directory.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
for _p in (str(PACKAGE_ROOT), str(TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The mean-P1D table columns expected by Pk.read_from_picca (HDU 1).
MEAN_COLUMNS = [
    "index_zbin", "zbin", "meank", "meanPk", "meanPk_raw", "meanPk_noise",
    "meanPk_diff", "errorPk", "errorPk_raw", "errorPk_noise", "errorPk_diff",
    "meanDelta2", "errorDelta2", "meanforest_snr", "meanforest_z",
    "meancor_reso", "errorcor_reso",
]


def make_picca_pk_fits(
    path, zbins=(2.2, 2.4, 2.6), nk=10, velunits=True, nqso=1000, with_cov=True
):
    """Write a synthetic picca-format mean-P1D FITS file.

    The file has the three HDUs consumed by :meth:`p1desi.pk_io.Pk.read_from_picca`:
    the per-(z, k) mean table, the per-z metadata (carrying ``VELUNITS`` / ``NQSO``
    in its header) and, optionally, a diagonal covariance.

    Returns:
        tuple: ``(np.array(zbins), k)`` — the bins and the shared k-grid used.
    """
    from astropy.io import fits
    from astropy.table import Table

    k = np.linspace(0.01, 0.1, nk)
    mean = {c: [] for c in MEAN_COLUMNS}
    meta = {c: [] for c in ["N_chunks", "k_min", "k_max", "z_min", "z_max"]}
    cov = {c: [] for c in
           ["index_zbin", "k1", "k2", "covariance",
            "boot_covariance", "boot_average_covariance"]}

    for i, z in enumerate(zbins):
        p = 0.1 / (1 + k)
        err = p * 0.05
        for j in range(nk):
            mean["index_zbin"].append(i)
            mean["zbin"].append(z)
            mean["meank"].append(k[j])
            mean["meanPk"].append(p[j])
            mean["meanPk_raw"].append(p[j] * 1.2)
            mean["meanPk_noise"].append(p[j] * 0.20)
            mean["meanPk_diff"].append(p[j] * 0.19)
            mean["errorPk"].append(err[j])
            mean["errorPk_raw"].append(err[j])
            mean["errorPk_noise"].append(p[j] * 0.02)
            mean["errorPk_diff"].append(p[j] * 0.02)
            mean["meanDelta2"].append(k[j] * p[j] / np.pi)
            mean["errorDelta2"].append(k[j] * err[j] / np.pi)
            mean["meanforest_snr"].append(3.0)
            mean["meanforest_z"].append(z)
            mean["meancor_reso"].append(1.0)
            mean["errorcor_reso"].append(0.01)
        meta["N_chunks"].append(100)
        meta["k_min"].append(float(k.min()))
        meta["k_max"].append(float(k.max()))
        meta["z_min"].append(z - 0.1)
        meta["z_max"].append(z + 0.1)
        if with_cov:
            K1, K2 = np.meshgrid(k, k)
            covm = np.diag(err ** 2)
            cov["index_zbin"].extend([i] * (nk * nk))
            cov["k1"].extend(K1.ravel())
            cov["k2"].extend(K2.ravel())
            cov["covariance"].extend(covm.ravel())
            cov["boot_covariance"].extend(covm.ravel())
            cov["boot_average_covariance"].extend(covm.ravel())

    t_mean = Table(mean)
    t_meta = Table(meta)
    t_meta.meta["VELUNITS"] = velunits
    t_meta.meta["NQSO"] = nqso
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(t_mean, name="MEANPK"),
        fits.BinTableHDU(t_meta, name="METADATA"),
    ]
    if with_cov:
        hdus.append(fits.BinTableHDU(Table(cov), name="COV"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return np.array(zbins), k


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run a test inside an isolated temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def picca_pk_file(tmp_path):
    """Path to a freshly written synthetic picca mean-P1D FITS file."""
    path = tmp_path / "mean_Pk1d_vel.fits"
    make_picca_pk_fits(str(path))
    return str(path)


@pytest.fixture
def pk(picca_pk_file):
    """A :class:`Pk` read from the synthetic picca file (velocity units)."""
    from p1desi import pk_io

    return pk_io.Pk.read_from_picca(picca_pk_file)
