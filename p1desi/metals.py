"""Metal-absorption contamination modelling for the DESI Lyman-alpha P1D.

This module fits and plots the oscillatory metal-contamination signal seen
in the "side band" regions of the DESI 1D power spectrum (P1D) measured
from `picca` outputs. Side band 1 (SB1) and side band 2 (SB2) are pixel
regions free of Lyman-alpha forest absorption but that still contain
correlated metal absorption (chiefly the Si IV and C IV doublets), whose
signature is an oscillation in k with a period set by the doublet velocity
(or wavelength) separation superimposed on a power-law continuum.

The module provides:
    - Parametric models (`model_SB1`, `model_SB2` and their per-redshift-bin
      rescaled variants) describing the side-band power spectrum as a
      power-law plus one (SB2) or two (SB1) damped cosine oscillations
      anchored on the C IV / Si IV doublet wavenumbers (`utils.kc4`,
      `utils.ksi4`, or their velocity-unit equivalents).
    - Fitting routines (`fit_model_SB1`, `fit_model_SB2`,
      `fit_model_SB1_indiv`, `fit_mean_side_band_rest`,
      `fit_indiv_side_band_1`) that use `scipy.optimize.curve_fit` to
      constrain these models on the mean and per-redshift-bin side-band
      power spectra.
    - Helper and plotting routines (`init_side_band_power`,
      `plot_side_band_fit`, `plot_side_band_fit_bis`) that build the
      rest-frame-rescaled mean side-band spectra and visualise the fits.
    - `fit_and_plot_side_band`, an orchestrator that runs the full
      mean + individual side-band fit and produces the diagnostic figure.

Wavenumbers `k` are expressed either in Angstrom^-1 (observed or rest
wavelength units) or in s/km (velocity units), matching the `velunits`
convention used throughout `p1desi`. Redshift `z` refers to the picca
`Pk1D` redshift bin.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from p1desi import pk_io, utils


def model_SB1(k, A, gamma, B, b, k1, phi1, C, c, k2, phi2):
    """Power-law-plus-two-oscillations model for side band 1 (SB1).

    SB1 contains two metal-doublet oscillations (e.g. Si IV and C IV), each
    modelled as an exponentially damped cosine of characteristic wavenumber
    k1 and k2, superimposed on a power-law continuum.

    Args:
        k (array-like): Wavenumber(s), in Angstrom^-1 or s/km.
        A (float): Power-law amplitude.
        gamma (float): Power-law index (model uses ``k ** (-gamma)``).
        B (float): Amplitude of the first damped oscillation.
        b (float): Damping rate of the first oscillation.
        k1 (float): Characteristic wavenumber of the first oscillation.
        phi1 (float): Phase offset of the first oscillation.
        C (float): Amplitude of the second damped oscillation.
        c (float): Damping rate of the second oscillation.
        k2 (float): Characteristic wavenumber of the second oscillation.
        phi2 (float): Phase offset of the second oscillation.

    Returns:
        numpy.ndarray: Model power spectrum evaluated at ``k``.
    """
    power_law = (
        A * k ** (-gamma)
        + B * np.exp(-b * k) * np.cos(2 * np.pi * ((k - k1) / k1) + phi1)
        + C * np.exp(-c * k) * np.cos(2 * np.pi * ((k - k2) / k2) + phi2)
    )
    return power_law


def model_SB2(k, A, gamma, B, b, k1, phi1):
    """Power-law-plus-one-oscillation model for side band 2 (SB2).

    SB2 contains a single metal-doublet oscillation, modelled as an
    exponentially damped cosine of characteristic wavenumber k1,
    superimposed on a power-law continuum.

    Args:
        k (array-like): Wavenumber(s), in Angstrom^-1 or s/km.
        A (float): Power-law amplitude.
        gamma (float): Power-law index (model uses ``k ** (-gamma)``).
        B (float): Amplitude of the damped oscillation.
        b (float): Damping rate of the oscillation.
        k1 (float): Characteristic wavenumber of the oscillation.
        phi1 (float): Phase offset of the oscillation.

    Returns:
        numpy.ndarray: Model power spectrum evaluated at ``k``.
    """
    power_law = A * k ** (-gamma) + B * np.exp(-b * k) * np.cos(
        2 * np.pi * ((k - k1) / k1) + phi1
    )
    return power_law


def model_SB1_indiv(param_mean, redshift, A, B, k):
    """Rescale the mean SB1 model to a single redshift bin (wavelength units).

    Evaluates the mean SB1 model (`model_SB1`) at the rest-frame wavenumber
    ``k * (1 + redshift)`` and applies a per-redshift-bin linear rescaling
    ``(A * k + B)``.

    Args:
        param_mean (sequence): Best-fit parameters of the mean `model_SB1`.
        redshift (float): Redshift of the bin being rescaled.
        A (float): Slope of the linear rescaling.
        B (float): Offset of the linear rescaling.
        k (array-like): Observed-frame wavenumber(s), in Angstrom^-1.

    Returns:
        numpy.ndarray: Rescaled model power spectrum evaluated at ``k``.
    """
    mean_p = model_SB1(k * (1 + redshift), *param_mean)
    return (A * k + B) * mean_p


def model_SB1_indiv_kms(param_mean, A, B, k):
    """Rescale the mean SB1 model to a single redshift bin (velocity units).

    Same as `model_SB1_indiv` but for velocity-unit wavenumbers (s/km),
    where no ``(1 + z)`` rest-frame rescaling of k is required.

    Args:
        param_mean (sequence): Best-fit parameters of the mean `model_SB1`.
        A (float): Slope of the linear rescaling.
        B (float): Offset of the linear rescaling.
        k (array-like): Wavenumber(s), in s/km.

    Returns:
        numpy.ndarray: Rescaled model power spectrum evaluated at ``k``.
    """
    mean_p = model_SB1(k, *param_mean)
    return (A * k + B) * mean_p


def fit_model_SB1(x, y, dy, k1, k2, dk, xmin=None, xmax=None):
    """Fit `model_SB1` to a side-band power spectrum via chi-square minimisation.

    NaN entries are removed and the fit is restricted to ``[xmin, xmax]``
    (defaulting to the data range). The oscillation wavenumbers are seeded
    at ``k1``/``k2`` and bounded to within ``+/- dk`` of those values.

    Args:
        x (numpy.ndarray): Wavenumber values.
        y (numpy.ndarray): Power spectrum values.
        dy (numpy.ndarray): Power spectrum uncertainties.
        k1 (float): Expected characteristic wavenumber of the first oscillation.
        k2 (float): Expected characteristic wavenumber of the second oscillation.
        dk (float): Half-width of the bound placed around ``k1`` and ``k2``.
        xmin (float, optional): Lower fit bound in x. Defaults to ``x.min()``.
        xmax (float, optional): Upper fit bound in x. Defaults to ``x.max()``.

    Returns:
        tuple: ``(param, cov)`` — best-fit `model_SB1` parameters and their
        covariance matrix, as returned by `scipy.optimize.curve_fit`.
    """
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy = x[mask], y[mask], dy[mask]
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    mask = (x >= xmin) & (x <= xmax)
    param, cov = curve_fit(
        model_SB1,
        xdata=x[mask],
        ydata=y[mask],
        sigma=dy[mask],
        p0=[1.0, 1.0, 0.02, 1.0, k1, 0.0, 0.01, 1.0, k2, 0.0],
        bounds=(
            [
                -np.inf,
                -np.inf,
                0.0,
                0.0,
                k1 - dk,
                -2 * np.pi,
                0.0,
                0.0,
                k2 - dk,
                -2 * np.pi,
            ],
            [
                np.inf,
                np.inf,
                0.5,
                np.inf,
                k1 + dk,
                2 * np.pi,
                0.1,
                np.inf,
                k2 + dk,
                2 * np.pi,
            ],
        ),
    )
    return param, cov


def fit_model_SB2(x, y, dy, k1, dk, xmin=None, xmax=None):
    """Fit `model_SB2` to a side-band power spectrum via chi-square minimisation.

    NaN entries are removed and the fit is restricted to ``[xmin, xmax]``
    (defaulting to the data range). The oscillation wavenumber is seeded
    at ``k1`` and bounded to within ``+/- dk`` of that value.

    Args:
        x (numpy.ndarray): Wavenumber values.
        y (numpy.ndarray): Power spectrum values.
        dy (numpy.ndarray): Power spectrum uncertainties.
        k1 (float): Expected characteristic wavenumber of the oscillation.
        dk (float): Half-width of the bound placed around ``k1``.
        xmin (float, optional): Lower fit bound in x. Defaults to ``x.min()``.
        xmax (float, optional): Upper fit bound in x. Defaults to ``x.max()``.

    Returns:
        tuple: ``(param, cov)`` — best-fit `model_SB2` parameters and their
        covariance matrix, as returned by `scipy.optimize.curve_fit`.
    """
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy = x[mask], y[mask], dy[mask]
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    mask = (x >= xmin) & (x <= xmax)
    param, cov = curve_fit(
        model_SB2,
        xdata=x[mask],
        ydata=y[mask],
        sigma=dy[mask],
        p0=[1.0, 1.0, 0.02, 1.0, k1, 0.0],
        bounds=(
            [-np.inf, -np.inf, 0.0, 0.0, k1 - dk, -2 * np.pi],
            [np.inf, np.inf, 0.5, np.inf, k1 + dk, 2 * np.pi],
        ),
    )
    return param, cov


def fit_model_SB1_indiv(
    param_mean, redshift, x, y, dy, xmin=None, xmax=None, velunits=False
):
    """Fit the per-redshift-bin linear rescaling (A, B) of the mean SB1 model.

    Holds the mean SB1 shape (`param_mean`) fixed and fits only the linear
    rescaling parameters A and B of `model_SB1_indiv` (or
    `model_SB1_indiv_kms` if ``velunits`` is True) to the data of a single
    redshift bin. NaN entries are removed and the fit is restricted to
    ``[xmin, xmax]`` (defaulting to the data range).

    Args:
        param_mean (sequence): Best-fit parameters of the mean `model_SB1`.
        redshift (float): Redshift of the bin being fitted.
        x (numpy.ndarray): Wavenumber values.
        y (numpy.ndarray): Power spectrum values.
        dy (numpy.ndarray): Power spectrum uncertainties.
        xmin (float, optional): Lower fit bound in x. Defaults to ``x.min()``.
        xmax (float, optional): Upper fit bound in x. Defaults to ``x.max()``.
        velunits (bool, optional): If True, fit in velocity units (s/km)
            using `model_SB1_indiv_kms` instead of `model_SB1_indiv`.
            Defaults to False.

    Returns:
        tuple: ``(param, cov)`` — best-fit ``(A, B)`` parameters and their
        covariance matrix, as returned by `scipy.optimize.curve_fit`.
    """
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy = x[mask], y[mask], dy[mask]
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    mask = (x >= xmin) & (x <= xmax)

    def model_to_fit(k, A, B):
        """Wrap `model_SB1_indiv`/`model_SB1_indiv_kms` for `curve_fit`.

        Args:
            k (array-like): Wavenumber(s) at which to evaluate the model.
            A (float): Slope of the linear rescaling.
            B (float): Offset of the linear rescaling.

        Returns:
            numpy.ndarray: Model power spectrum evaluated at ``k``, using
            `model_SB1_indiv_kms` if ``velunits`` is True, else
            `model_SB1_indiv`.
        """
        if velunits:
            return model_SB1_indiv_kms(param_mean, A, B, k)
        else:
            return model_SB1_indiv(param_mean, redshift, A, B, k)

    param, cov = curve_fit(
        model_to_fit,
        xdata=x[mask],
        ydata=y[mask],
        sigma=dy[mask],
        p0=[0.0, 0.0],
        bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
    )
    return param, cov


def init_side_band_power(pSB1, pSB2, zmax):
    """Build rest-frame-rescaled mean SB1/SB2 power spectra across redshift bins.

    For each redshift bin below ``zmax``, rescales the SB1 and SB2
    wavenumbers to the rest frame (``k * (1 + z)``, or left as-is if
    ``pSB1.velunits`` is True), interpolates each bin's power spectrum and
    error onto a common rest-frame k grid, and averages over redshift bins.

    Args:
        pSB1: Side-band-1 `Pk1D`-like power-spectrum object with attributes
            ``zbin``, ``k`` (dict keyed by z), ``p``, ``err``, ``velunits``.
        pSB2: Side-band-2 power-spectrum object with the same interface as
            ``pSB1``.
        zmax (float): Upper redshift bound (exclusive); only bins with
            ``z < zmax`` are included.

    Returns:
        dict: Mapping with keys ``"krescaleSB1"``, ``"krescaleSB2"`` (common
        rest-frame k grids), ``"pkrestSB1"``, ``"pkrestSB2"`` (redshift-mean
        power spectra) and ``"errorpkrestSB1"``, ``"errorpkrestSB2"``
        (standard error on the mean).
    """

    velunits = pSB1.velunits

    kSB1, kSB2 = [], []
    krestSB1, krestSB2 = [], []
    pkrestSB1, pkrestSB2 = [], []
    errorpkrestSB1, errorpkrestSB2 = [], []

    for z in pSB1.zbin:
        if z < zmax:
            krestSB1.append(pSB1.k[z] * (1 + z))
            krestSB2.append(pSB2.k[z] * (1 + z))
            kSB1.append(pSB1.k[z])
            kSB2.append(pSB2.k[z])

    if velunits:
        krestSB1 = np.array(kSB1)
        krestSB2 = np.array(kSB2)

    else:
        krestSB1 = np.array(krestSB1)
        krestSB2 = np.array(krestSB2)

    krescaleSB1 = np.linspace(
        np.nanmin(krestSB1), np.nanmax(krestSB1), krestSB1.shape[1]
    )
    krescaleSB2 = np.linspace(
        np.nanmin(krestSB2), np.nanmax(krestSB2), krestSB2.shape[1]
    )
    for i, z in enumerate(pSB1.zbin):
        if z < zmax:
            pkrestSB1.append(np.interp(krescaleSB1, krestSB1[i], pSB1.p[z]))
            errorpkrestSB1.append(np.interp(krescaleSB1, krestSB1[i], pSB1.err[z]))
            pkrestSB2.append(np.interp(krescaleSB2, krestSB2[i], pSB2.p[z]))
            errorpkrestSB2.append(np.interp(krescaleSB2, krestSB2[i], pSB2.err[z]))

    mean_dict = {}
    mean_dict["krescaleSB1"] = krescaleSB1
    mean_dict["krescaleSB2"] = krescaleSB2
    mean_dict["pkrestSB1"] = np.nanmean(pkrestSB1, axis=0)
    mean_dict["pkrestSB2"] = np.nanmean(pkrestSB2, axis=0)
    mean_dict["errorpkrestSB1"] = np.nanmean(errorpkrestSB1, axis=0) / np.sqrt(
        len(pkrestSB1)
    )
    mean_dict["errorpkrestSB2"] = np.nanmean(errorpkrestSB2, axis=0) / np.sqrt(
        len(pkrestSB2)
    )

    return mean_dict


def fit_mean_side_band_rest(
    mean_dict, dkrest=1.0, kminrest=None, kmaxrest=8, velunits=False
):
    """Fit the mean rest-frame SB1 and SB2 power spectra.

    Fits `model_SB1` to the mean SB1 spectrum (anchored on the C IV and
    Si IV doublet wavenumbers) and `model_SB2` to the mean SB2 spectrum
    (anchored on the C IV doublet wavenumber), using the arrays produced by
    `init_side_band_power`.

    Args:
        mean_dict (dict): Output of `init_side_band_power`, containing
            ``"krescaleSB1"``, ``"pkrestSB1"``, ``"errorpkrestSB1"``,
            ``"krescaleSB2"``, ``"pkrestSB2"``, ``"errorpkrestSB2"``.
        dkrest (float, optional): Half-width of the bound placed around the
            doublet wavenumbers. Defaults to 1.0.
        kminrest (float, optional): Lower fit bound in rest-frame k.
            Defaults to None (use the data minimum).
        kmaxrest (float, optional): Upper fit bound in rest-frame k.
            Defaults to 8.
        velunits (bool, optional): If True, use the velocity-unit doublet
            wavenumbers (`utils.kc4_speed`, `utils.ksi4_speed`); otherwise
            use the wavelength-unit ones (`utils.kc4`, `utils.ksi4`).
            Defaults to False.

    Returns:
        tuple: ``(param_SB1_mean, param_SB2_mean, cov_SB1_mean,
        cov_SB2_mean)`` — best-fit parameters and covariance matrices for
        the mean SB1 and SB2 models.
    """
    if velunits:
        kc4 = utils.kc4_speed
        ksi4 = utils.ksi4_speed
    else:
        kc4 = utils.kc4
        ksi4 = utils.ksi4

    (param_SB1_mean, cov_SB1_mean) = fit_model_SB1(
        mean_dict["krescaleSB1"],
        mean_dict["pkrestSB1"],
        mean_dict["errorpkrestSB1"],
        kc4,
        ksi4,
        dkrest,
        xmin=kminrest,
        xmax=kmaxrest,
    )

    (param_SB2_mean, cov_SB2_mean) = fit_model_SB2(
        mean_dict["krescaleSB2"],
        mean_dict["pkrestSB2"],
        mean_dict["errorpkrestSB2"],
        kc4,
        dkrest,
        xmin=kminrest,
        xmax=kmaxrest,
    )
    return param_SB1_mean, param_SB2_mean, cov_SB1_mean, cov_SB2_mean


def fit_indiv_side_band_1(pSB1, zmax, param_SB1_mean, kmin=None, kmax=8, save_fit=None):
    """Fit the per-redshift-bin linear rescaling of SB1 for every bin below zmax.

    Loops over ``pSB1.zbin`` and, for each bin with ``z < zmax``, calls
    `fit_model_SB1_indiv` to fit the (A, B) rescaling of the mean SB1 model
    to that bin's observed-frame power spectrum. Optionally pickles the
    mean and individual fit parameters.

    Args:
        pSB1: Side-band-1 power-spectrum object with attributes ``zbin``,
            ``k`` (dict keyed by z), ``p``, ``err``, ``velunits``.
        zmax (float): Upper redshift bound (exclusive); only bins with
            ``z < zmax`` are fitted.
        param_SB1_mean (sequence): Best-fit parameters of the mean SB1
            model (from `fit_mean_side_band_rest`).
        kmin (float, optional): Lower fit bound in k. Defaults to None (use
            the data minimum).
        kmax (float, optional): Upper fit bound in k. Defaults to 8.
        save_fit (str, optional): If given, path to pickle
            ``(param_SB1_mean, param_SB1_indiv)`` to. Defaults to None (no
            saving).

    Returns:
        tuple: ``(param_SB1_indiv, cov_SB1_indiv)`` — lists of best-fit
        (A, B) parameters and covariance matrices, one entry per redshift
        bin below ``zmax``.
    """
    param_SB1_indiv = []
    cov_SB1_indiv = []
    for z in pSB1.zbin:
        if z < zmax:
            (param_SB1, cov_SB1) = fit_model_SB1_indiv(
                param_SB1_mean,
                z,
                pSB1.k[z],
                pSB1.p[z],
                pSB1.err[z],
                xmin=kmin,
                xmax=kmax,
                velunits=pSB1.velunits,
            )
            param_SB1_indiv.append(param_SB1)
            cov_SB1_indiv.append(cov_SB1)
    if save_fit is not None:
        pickle.dump((param_SB1_mean, param_SB1_indiv), open(save_fit, "wb"))
    return param_SB1_indiv, cov_SB1_indiv


def plot_side_band_fit(
    name_out,
    plot_P,
    mean_dict,
    pSB1,
    zmax,
    param_SB1_mean,
    param_SB2_mean,
    param_SB1_indiv,
    nb_bins,
    kmaxrest,
    kmax,
    save_txt=None,
    **kwargs,
):
    """Plot the mean SB1/SB2 rest-frame fits and the individual SB1 redshift-bin fits.

    Produces a two-panel figure: the top panel shows the mean SB1 and SB2
    rest-frame power spectra with their `model_SB1`/`model_SB2` fits; the
    bottom panel shows, for each redshift bin below ``zmax``, the observed
    SB1 power spectrum with its individually rescaled fit
    (`model_SB1_indiv`/`model_SB1_indiv_kms`). Saves the figure to
    ``name_out`` and, if requested, the underlying arrays to text files.

    Args:
        name_out (str): Output path for the saved figure.
        plot_P (bool): If True, plot P(k) directly; if False, plot the
            dimensionless ``k * P(k) / pi`` quantity instead.
        mean_dict (dict): Output of `init_side_band_power`.
        pSB1: Side-band-1 power-spectrum object with attributes ``zbin``,
            ``k``, ``p``, ``err``, ``norm_p``, ``norm_err``,
            ``number_chunks``, ``velunits``.
        zmax (float): Upper redshift bound (exclusive) for the bottom panel.
        param_SB1_mean (sequence): Best-fit mean SB1 model parameters.
        param_SB2_mean (sequence): Best-fit mean SB2 model parameters.
        param_SB1_indiv (sequence): Per-redshift-bin (A, B) fit parameters
            from `fit_indiv_side_band_1`.
        nb_bins (int): Number of points used to sample the fitted model
            curves.
        kmaxrest (float or None): Upper x-limit / fit-curve bound for the
            top (rest-frame) panel. If None, taken from
            ``mean_dict["krescaleSB1"]``.
        kmax (float): Upper x-limit / fit-curve bound for the bottom
            (observed-frame) panel.
        save_txt (str, optional): If given, path prefix used to save the
            mean SB1, mean SB2, and individual SB1 arrays to
            ``{save_txt}_1.txt``, ``{save_txt}_2.txt``, ``{save_txt}_3.txt``.
            Defaults to None (no saving).
        **kwargs: Plot styling overrides, including ``style``, ``figsize``,
            ``ncol_legend``, ``size_legend``, ``fontsize``, ``size``,
            ``markersize``, ``ylim``, ``ylabel_1``, ``ylabel_2``,
            ``xlabel_1``, ``xlabel_2``.

    Returns:
        None
    """
    style = utils.return_key(kwargs, "style", None)
    if style is not None:
        plt.style.use(style)

    figsize = utils.return_key(kwargs, "figsize", (9, 9))
    ncol_legend = utils.return_key(kwargs, "ncol_legend", 2)
    size_legend = utils.return_key(kwargs, "size_legend", 11)
    fontsize = utils.return_key(kwargs, "fontsize", 15)
    size = utils.return_key(kwargs, "size", 14)
    markersize = utils.return_key(kwargs, "markersize", 8)

    ylim = utils.return_key(kwargs, "ylim", [-0.01, 0.05] if plot_P else [-0.001, 0.02])
    ylabel_1 = utils.return_key(
        kwargs,
        "ylabel_1",
        (
            r"$P_{\mathrm{SB}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$"
            if pSB1.velunits
            else r"$P_{\mathrm{SB}}~[\AA]$"
        ),
    )
    ylabel_2 = utils.return_key(
        kwargs,
        "ylabel_2",
        (
            r"$P_{\mathrm{SB1}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$"
            if pSB1.velunits
            else r"$P_{\mathrm{SB1}}~[\AA]$"
        ),
    )
    xlabel_1 = utils.return_key(
        kwargs,
        "xlabel_1",
        (
            r"$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$"
            if pSB1.velunits
            else r"$k_{\mathrm{rest}}=k_{\mathrm{obs}}(1+z)~[\mathrm{\AA}^{-1}]$"
        ),
    )
    xlabel_2 = utils.return_key(
        kwargs,
        "xlabel_2",
        (
            r"$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$"
            if pSB1.velunits
            else r"$k_{\mathrm{obs}}~[\mathrm{\AA}^{-1}]$"
        ),
    )

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=False)

    if kmaxrest is None:
        kmaxrest = np.max(mean_dict["krescaleSB1"])

    k_fit_mean_rest = np.linspace(np.min(mean_dict["krescaleSB1"]), kmaxrest, nb_bins)
    if plot_P:
        p_plot_SB1 = mean_dict["pkrestSB1"]
        err_plot_SB1 = mean_dict["errorpkrestSB1"]
        p_plot_fit_SB1 = model_SB1(k_fit_mean_rest, *param_SB1_mean)
        p_plot_SB2 = mean_dict["pkrestSB2"]
        err_plot_SB2 = mean_dict["errorpkrestSB2"]
        p_plot_fit_SB2 = model_SB2(k_fit_mean_rest, *param_SB2_mean)
    else:
        p_plot_SB1 = mean_dict["krescaleSB1"] * mean_dict["pkrestSB1"] / np.pi
        err_plot_SB1 = mean_dict["krescaleSB1"] * mean_dict["errorpkrestSB1"] / np.pi
        p_plot_fit_SB1 = (
            k_fit_mean_rest * model_SB1(k_fit_mean_rest, *param_SB1_mean) / np.pi
        )
        p_plot_SB2 = mean_dict["krescaleSB2"] * mean_dict["pkrestSB2"] / np.pi
        err_plot_SB2 = mean_dict["krescaleSB2"] * mean_dict["errorpkrestSB2"] / np.pi
        p_plot_fit_SB2 = (
            k_fit_mean_rest * model_SB2(k_fit_mean_rest, *param_SB2_mean) / np.pi
        )

    ax[0].errorbar(
        mean_dict["krescaleSB1"],
        p_plot_SB1,
        err_plot_SB1,
        marker=".",
        ls="None",
        markersize=markersize,
    )
    ax[0].plot(k_fit_mean_rest, p_plot_fit_SB1, color="C0")

    ax[0].errorbar(
        mean_dict["krescaleSB2"],
        p_plot_SB2,
        err_plot_SB2,
        marker=".",
        ls="None",
        markersize=markersize,
    )
    ax[0].plot(k_fit_mean_rest, p_plot_fit_SB2, color="C1")

    color = cm.rainbow(np.linspace(0, 1, len(pSB1.zbin[pSB1.zbin < zmax])))

    z_arr, k_arr, p_arr, err_arr = [], [], [], []
    for i, z in enumerate(pSB1.zbin):
        if z < zmax:
            k_fit = np.linspace(np.nanmin(pSB1.k[z]), kmax, nb_bins)
            if pSB1.velunits:
                model_fitted = model_SB1_indiv_kms(
                    param_SB1_mean, param_SB1_indiv[i][0], param_SB1_indiv[i][1], k_fit
                )
            else:
                model_fitted = model_SB1_indiv(
                    param_SB1_mean,
                    z,
                    param_SB1_indiv[i][0],
                    param_SB1_indiv[i][1],
                    k_fit,
                )
            if plot_P:
                p_plot = pSB1.p[z]
                err_plot = pSB1.err[z]
                p_plot_fit = model_fitted
            else:
                p_plot = pSB1.norm_p[z]
                err_plot = pSB1.norm_err[z]
                p_plot_fit = k_fit * model_fitted / np.pi
            ax[1].errorbar(
                pSB1.k[z],
                p_plot,
                err_plot,
                label=f"z = {z:.1f} ({pSB1.number_chunks[z]} chunks)",
                marker=".",
                ls="None",
                color=color[i],
                markersize=markersize,
            )
            ax[1].plot(k_fit, p_plot_fit, color=color[i])

            z_arr.append([z for i in range(len(pSB1.k[z]))])
            k_arr.append(pSB1.k[z])
            p_arr.append(p_plot)
            err_arr.append(err_plot)

    z_arr, k_arr, p_arr, err_arr = (
        np.concatenate(z_arr),
        np.concatenate(k_arr),
        np.concatenate(p_arr),
        np.concatenate(err_arr),
    )

    ax[1].set_ylim(ylim)
    ax[0].set_ylim(ylim)
    ax[1].set_xlim([0, kmax])
    ax[0].set_xlim([0, kmaxrest])

    ax[0].set_ylabel(ylabel_1, fontsize=fontsize)
    ax[0].set_xlabel(xlabel_1, fontsize=fontsize)
    ax[0].tick_params("x", labelsize=size)
    ax[0].tick_params("y", labelsize=size)

    ax[1].legend(ncol=ncol_legend, fontsize=size_legend)
    ax[1].set_ylabel(ylabel_2, fontsize=fontsize)
    ax[1].set_xlabel(xlabel_2, fontsize=fontsize)
    ax[1].tick_params("x", labelsize=size)
    ax[1].tick_params("y", labelsize=size)

    legend_elements = [
        Line2D([], [], color="C0", marker=None, linestyle="-", label="SB1"),
        Line2D([], [], color="C1", marker=None, linestyle="-", label="SB2"),
    ]
    ax[0].legend(handles=legend_elements, fontsize=size_legend)

    plt.tight_layout()
    plt.savefig(name_out)

    if save_txt is not None:
        np.savetxt(
            f"{save_txt}_1.txt",
            np.transpose(
                np.stack([mean_dict["krescaleSB1"], p_plot_SB1, err_plot_SB1])
            ),
            header="SB1 WAVENUMBER [Ang^-1] & SB1 POWER SPECTRUM & SB1 ERROR",
        )

        np.savetxt(
            f"{save_txt}_2.txt",
            np.transpose(
                np.stack([mean_dict["krescaleSB2"], p_plot_SB2, err_plot_SB2])
            ),
            header="SB2 WAVENUMBER [Ang^-1] & SB2 POWER SPECTRUM & SB2 ERROR",
        )

        np.savetxt(
            f"{save_txt}_3.txt",
            np.transpose(np.stack([z_arr, k_arr, p_arr, err_arr])),
            header="REDSHIFT & SB1 INDIV WAVENUMBER [Ang^-1] & SB1 INDIV POWER SPECTRUM & SB1 INDIV ERROR",
        )


def plot_side_band_fit_bis(
    name_out,
    plot_P,
    pSB1,
    zmax,
    param_SB1_mean,
    param_SB1_indiv,
    nb_bins,
    kmax,
    kmin,
    save_txt=None,
    **kwargs,
):
    """Plot the individual SB1 redshift-bin fits in a single panel.

    Variant of `plot_side_band_fit` that omits the mean SB1/SB2 rest-frame
    panel and only plots, for each redshift bin below ``zmax``, the observed
    SB1 power spectrum with its individually rescaled fit
    (`model_SB1_indiv`/`model_SB1_indiv_kms`). Saves the figure to
    ``name_out`` and, if requested, the underlying arrays to a text file.

    Args:
        name_out (str): Output path for the saved figure.
        plot_P (bool): If True, plot P(k) directly; if False, plot the
            dimensionless ``k * P(k) / pi`` quantity instead.
        pSB1: Side-band-1 power-spectrum object with attributes ``zbin``,
            ``k``, ``p``, ``err``, ``norm_p``, ``norm_err``,
            ``number_chunks``, ``velunits``.
        zmax (float): Upper redshift bound (exclusive) for the bins plotted.
        param_SB1_mean (sequence): Best-fit mean SB1 model parameters.
        param_SB1_indiv (sequence): Per-redshift-bin (A, B) fit parameters
            from `fit_indiv_side_band_1`.
        nb_bins (int): Number of points used to sample the fitted model
            curves.
        kmax (float): Upper x-limit / fit-curve bound.
        kmin (float): Lower x-limit.
        save_txt (str, optional): If given, path used to save the redshift,
            k, power, and error arrays to ``{save_txt}.txt``. Defaults to
            None (no saving).
        **kwargs: Plot styling overrides, including ``style``, ``figsize``,
            ``ncol_legend``, ``size_legend``, ``fontsize``, ``size``,
            ``markersize``, ``ylim``, ``ylabel_2``, ``xlabel_2``.

    Returns:
        None
    """
    style = utils.return_key(kwargs, "style", None)
    if style is not None:
        plt.style.use(style)

    figsize = utils.return_key(kwargs, "figsize", (9, 9))
    ncol_legend = utils.return_key(kwargs, "ncol_legend", 2)
    size_legend = utils.return_key(kwargs, "size_legend", 11)
    fontsize = utils.return_key(kwargs, "fontsize", 15)
    size = utils.return_key(kwargs, "size", 14)
    markersize = utils.return_key(kwargs, "markersize", 8)

    ylim = utils.return_key(kwargs, "ylim", [-0.01, 0.05] if plot_P else [-0.001, 0.02])

    ylabel = utils.return_key(
        kwargs,
        "ylabel_2",
        (
            r"$P_{\mathrm{SB1}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$"
            if pSB1.velunits
            else r"$P_{\mathrm{SB1}}~[\AA]$"
        ),
    )
    xlabel = utils.return_key(
        kwargs,
        "xlabel_2",
        (
            r"$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$"
            if pSB1.velunits
            else r"$k~[\mathrm{\AA}^{-1}]$"
        ),
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=False)

    color = cm.rainbow(np.linspace(0, 1, len(pSB1.zbin[pSB1.zbin < zmax])))

    z_arr, k_arr, p_arr, err_arr = [], [], [], []
    for i, z in enumerate(pSB1.zbin):
        if z < zmax:
            k_fit = np.linspace(np.nanmin(pSB1.k[z]), kmax, nb_bins)
            if pSB1.velunits:
                model_fitted = model_SB1_indiv_kms(
                    param_SB1_mean,
                    param_SB1_indiv[i][0],
                    param_SB1_indiv[i][1],
                    k_fit,
                )
            else:
                model_fitted = model_SB1_indiv(
                    param_SB1_mean,
                    z,
                    param_SB1_indiv[i][0],
                    param_SB1_indiv[i][1],
                    k_fit,
                )
            if plot_P:
                p_plot = pSB1.p[z]
                err_plot = pSB1.err[z]
                p_plot_fit = model_fitted
            else:
                p_plot = pSB1.norm_p[z]
                err_plot = pSB1.norm_err[z]
                p_plot_fit = k_fit * model_fitted / np.pi
            ax.errorbar(
                pSB1.k[z],
                p_plot,
                err_plot,
                label=f"z = {z:.1f} ({pSB1.number_chunks[z]} chunks)",
                marker=".",
                ls="None",
                color=color[i],
                markersize=markersize,
            )
            ax.plot(k_fit, p_plot_fit, color=color[i])

            z_arr.append([z for i in range(len(pSB1.k[z]))])
            k_arr.append(pSB1.k[z])
            p_arr.append(p_plot)
            err_arr.append(err_plot)

    z_arr, k_arr, p_arr, err_arr = (
        np.concatenate(z_arr),
        np.concatenate(k_arr),
        np.concatenate(p_arr),
        np.concatenate(err_arr),
    )

    ax.set_ylim(ylim)
    ax.set_xlim([kmin, kmax])

    ax.legend(ncol=ncol_legend, fontsize=size_legend)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params("x", labelsize=size)
    ax.tick_params("y", labelsize=size)

    plt.tight_layout()
    plt.savefig(name_out)

    if save_txt is not None:
        np.savetxt(
            f"{save_txt}.txt",
            np.transpose(np.stack([z_arr, k_arr, p_arr, err_arr])),
            header="REDSHIFT & SB1 INDIV WAVENUMBER [Ang^-1] & SB1 INDIV POWER SPECTRUM & SB1 INDIV ERROR",
        )


def fit_and_plot_side_band(
    pSB1,
    pSB2,
    name_out,
    zmax,
    nb_bins,
    dkrest,
    kminrest,
    kmaxrest,
    kmin,
    kmax,
    plot_P=True,
    save_fit=None,
    save_txt=None,
    **plt_args,
):
    """Run the full side-band metal-contamination fit and produce the diagnostic plot.

    Orchestrates the complete pipeline: builds the mean rest-frame SB1/SB2
    spectra (`init_side_band_power`), fits the mean SB1 and SB2 models
    (`fit_mean_side_band_rest`), fits the per-redshift-bin SB1 rescaling
    (`fit_indiv_side_band_1`), and plots everything (`plot_side_band_fit`).

    Args:
        pSB1: Side-band-1 power-spectrum object with attributes ``zbin``,
            ``k``, ``p``, ``err``, ``norm_p``, ``norm_err``,
            ``number_chunks``, ``velunits``.
        pSB2: Side-band-2 power-spectrum object with the same interface as
            ``pSB1``.
        name_out (str): Output path for the saved figure.
        zmax (float): Upper redshift bound (exclusive) for included bins.
        nb_bins (int): Number of points used to sample the fitted model
            curves.
        dkrest (float): Half-width of the bound placed around the doublet
            wavenumbers in the mean rest-frame fit.
        kminrest (float or None): Lower rest-frame fit bound.
        kmaxrest (float): Upper rest-frame fit bound / plot x-limit.
        kmin (float): Lower observed-frame fit bound for the individual
            SB1 fits.
        kmax (float): Upper observed-frame fit bound / plot x-limit.
        plot_P (bool, optional): If True, plot P(k) directly; if False,
            plot the dimensionless ``k * P(k) / pi`` quantity. Defaults to
            True.
        save_fit (str, optional): If given, path to pickle the mean and
            individual SB1 fit parameters to. Defaults to None.
        save_txt (str, optional): If given, path prefix used to save the
            plotted arrays to text files. Defaults to None.
        **plt_args: Additional plot styling overrides forwarded to
            `plot_side_band_fit`.

    Returns:
        tuple: ``(param_SB1_mean, param_SB2_mean, cov_SB1_mean,
        cov_SB2_mean, param_SB1_indiv, cov_SB1_indiv)`` — mean and
        per-redshift-bin best-fit parameters and covariance matrices.
    """
    mean_dict = init_side_band_power(pSB1, pSB2, zmax)

    (
        param_SB1_mean,
        param_SB2_mean,
        cov_SB1_mean,
        cov_SB2_mean,
    ) = fit_mean_side_band_rest(
        mean_dict,
        dkrest=dkrest,
        kminrest=kminrest,
        kmaxrest=kmaxrest,
        velunits=pSB1.velunits,
    )

    (param_SB1_indiv, cov_SB1_indiv) = fit_indiv_side_band_1(
        pSB1, zmax, param_SB1_mean, kmin=kmin, kmax=kmax, save_fit=save_fit
    )

    plot_side_band_fit(
        name_out,
        plot_P,
        mean_dict,
        pSB1,
        zmax,
        param_SB1_mean,
        param_SB2_mean,
        param_SB1_indiv,
        nb_bins,
        kmaxrest,
        kmax,
        save_txt=save_txt,
        **plt_args,
    )
    return (
        param_SB1_mean,
        param_SB2_mean,
        cov_SB1_mean,
        cov_SB2_mean,
        param_SB1_indiv,
        cov_SB1_indiv,
    )
