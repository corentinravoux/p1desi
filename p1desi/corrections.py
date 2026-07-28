"""Assembly and application of P1D corrections for DESI Lyman-alpha analysis.

This module builds and applies the corrections needed to turn a raw picca
1D power spectrum estimate into a corrected P1D: masking/continuum
corrections (HCD, sky lines, continuum fitting, BAL, CCD resolution), noise
corrections, and side-band (metal) subtraction corrections. Corrections are
fit and evaluated as a function of wavenumber ``k`` (in Angstrom^-1, or in
s/km when ``pk.velunits`` is True) over a set of redshift bins, and can be
built either from DESI-derived power-spectrum ratios/fits or from legacy
eBOSS ASCII correction tables.
"""

import glob
import os
import pickle
from functools import partial

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import curve_fit

from p1desi import hcd, metals, utils

###################################################
############## Creating corrections ###############
###################################################


def plot_and_compute_ratio_power(
    pk,
    pk2,
    path_out,
    name_correction,
    suffix,
    model,
    zmax,
    kmin_AA,
    kmin_fit_AA,
    kmax_AA,
    model_param=None,
    use_covariance_fit=False,
    use_boot=False,
    **plt_args,
):
    """
    plot_and_compute_ratio_power - Plot the ratio of two power spectra and fit a curve to the data

    Inputs:

        pk: object containing the power spectrum data to plot, with attributes k, p, and err for the wave number k, power spectrum P, and error on P respectively
        pk2: object containing the power spectrum data to divide pk by, with attributes k, p, and err for the wave number k, power spectrum P, and error on P respectively
        path_out: string specifying the filepath to save the plot
        name_correction: string specifying the correction applied to the power spectra
        suffix: string to add to the end of the plot file name
        model: string specifying the type of curve to fit to the data, either "poly" for a polynomial or "rogers" for a function defined in Rogers et al. (2020)
        zmax: maximum redshift to plot
        kmin_AA: minimum wave number in inverse Angstroms to plot
        kmin_fit_AA: minimum wave number in inverse Angstroms to use for fitting the curve
        kmax_AA: maximum wave number in inverse Angstroms to plot
        model_param: dictionary of parameters for the fitting function, required if model is "poly" and should contain the key "order_poly" specifying the order of the polynomial
        **plt_args: additional arguments to be passed to the plot, including:
            dk: shift in wave number between redshift bins
            zseparation: redshift at which to split the plot into two subplots
            ymin: minimum y value for the plot
            ymax: maximum y value for the plot
            ylabel: string for the y-axis label

    Outputs:

        fig: figure object for the plot
        ax: axis object for the plot
        params: list of arrays of fitted curve parameters for each redshift
    """

    dk = utils.return_key(plt_args, "dk", 0.0)
    zseparation = utils.return_key(plt_args, "zseparation", 2.9)
    ymin = utils.return_key(plt_args, "ymin", 0.95)
    ymax = utils.return_key(plt_args, "ymax", 1.05)
    ylabel = utils.return_key(plt_args, "ylabel", r"$A_{\mathrm{corr}}$")
    marker_style = utils.return_key(plt_args, "marker_style", ".")
    marker_size = utils.return_key(plt_args, "marker_size", 7)
    fontsize = utils.return_key(plt_args, "fontsize", 21)
    fontsize_y = utils.return_key(plt_args, "fontsize_y", 21)
    fontlegend = utils.return_key(plt_args, "fontlegend", 18)
    fontticks = utils.return_key(plt_args, "fontticks", 19)
    color_map = utils.return_key(plt_args, "color_map", "default")
    if color_map == "default":
        colors = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        colors = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))
    elif color_map == "jet":
        colors = cm.jet(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    params = []

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    z_arr, k_arr, ratio_arr, err_arr = [], [], [], []
    for i, z in enumerate(pk.zbin):
        if z < zseparation:
            axplt = ax[0]
        else:
            axplt = ax[1]
        if z < zmax:
            if pk.velunits:
                kmax = float(utils.kAAtokskm(kmax_AA, z=z))
                kmin = float(utils.kAAtokskm(kmin_AA, z=z))
            else:
                kmax = kmax_AA
                kmin = kmin_AA
            k = pk.k[z]

            pk2_interp = interp1d(
                pk2.k[z],
                pk2.p[z],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(k)
            err_pk2_interp = interp1d(
                pk2.k[z],
                pk2.err[z],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )(k)

            ratio = pk.p[z] / pk2_interp
            err_ratio = (pk.p[z] / pk2_interp) * np.sqrt(
                (pk.err[z] / pk.p[z]) ** 2 + (err_pk2_interp / pk2_interp) ** 2
            )

            mask = (
                np.isnan(k)
                | np.isnan(ratio)
                | np.isnan(err_ratio)
                | (k > kmax)
                | (k < kmin)
            )
            k = k[~mask]
            ratio = ratio[~mask]
            err_ratio = err_ratio[~mask]

            axplt.errorbar(
                k + i * dk,
                ratio,
                yerr=err_ratio,
                marker=marker_style,
                color=colors[i],
                markersize=marker_size,
                linestyle="None",
                label=r"$z = ${:1.1f}".format(z),
            )
            z_arr.append([z for i in range(len(k))])
            k_arr.append(k)
            ratio_arr.append(ratio)
            err_arr.append(err_ratio)

            if model is not None:
                if use_covariance_fit:

                    if use_boot:
                        cov1 = pk.boot_cov[z]
                        cov2 = pk2.boot_cov[z]
                    else:
                        cov1 = pk.cov[z]
                        cov2 = pk2.cov[z]

                    k11 = pk.cov_k1[z]
                    k12 = pk.cov_k2[z]
                    mask_cov1 = (k11 < kmax) & (k12 < kmax)
                    mask_cov1 &= (k11 > kmin) & (k12 > kmin)

                    k11 = k11[mask_cov1]
                    k12 = k12[mask_cov1]
                    cov1 = cov1[mask_cov1]

                    k11_matrix = k11.reshape(
                        int(np.sqrt(k11.size)), int(np.sqrt(k11.size))
                    )
                    k12_matrix = k12.reshape(
                        int(np.sqrt(k12.size)), int(np.sqrt(k12.size))
                    )
                    cov1 = cov1.reshape(
                        int(np.sqrt(cov1.size)), int(np.sqrt(cov1.size))
                    )

                    k21 = pk2.cov_k1[z]
                    k22 = pk2.cov_k2[z]
                    mask_cov2 = (k21 < kmax) & (k22 < kmax)
                    mask_cov2 &= (k21 > kmin) & (k22 > kmin)

                    k21 = k21[mask_cov2]
                    k22 = k22[mask_cov2]
                    cov2 = cov2[mask_cov2]

                    k21_matrix = k21.reshape(
                        int(np.sqrt(k21.size)), int(np.sqrt(k21.size))
                    )
                    k22_matrix = k22.reshape(
                        int(np.sqrt(k22.size)), int(np.sqrt(k22.size))
                    )
                    cov2 = cov2.reshape(
                        int(np.sqrt(cov2.size)), int(np.sqrt(cov2.size))
                    )
                    K11, K12 = np.meshgrid(
                        k11_matrix[:, 0], k12_matrix[0, :], indexing="ij"
                    )

                    cov2_interp = RegularGridInterpolator(
                        (k21_matrix[:, 0], k22_matrix[0, :]), cov2, method="linear"
                    )((K11, K12))

                    dR_dX1 = 1 / pk2_interp[~mask]
                    dR_dX2 = pk.p[z][~mask] / (pk2_interp[~mask] ** 2)

                    dR_dX1_outer = np.outer(dR_dX1, dR_dX1)
                    dR_dX2_outer = np.outer(dR_dX2, dR_dX2)

                    cov_ratio = dR_dX1_outer * cov1 + dR_dX2_outer * cov2_interp
                    err_fit = cov_ratio
                else:
                    err_fit = err_ratio
            if model == "poly":

                def poly_func(x, *coefficients):
                    """Evaluate a polynomial with highest-degree coefficient first.

                    Args:
                        x (array_like): Wavenumber(s) at which to evaluate the polynomial.
                        *coefficients (float): Polynomial coefficients, ordered from the
                            highest degree term to the constant term.

                    Returns:
                        array_like: Polynomial value(s) evaluated at ``x``.
                    """
                    return np.sum(
                        coeff * (x ** (len(coefficients) - 1 - i))
                        for i, coeff in enumerate(coefficients)
                    )

                popt, _ = curve_fit(
                    poly_func,
                    xdata=k,
                    ydata=ratio,
                    sigma=err_fit,
                    p0=[1.0 for i in range(model_param["order_poly"] + 1)],
                )

                k_th = np.linspace(np.min(k), np.max(k), 2000)
                axplt.plot(
                    k_th,
                    poly_func(k_th, *popt),
                    color=colors[i],
                    ls="--",
                )
                params.append(popt)

            elif model == "power":
                popt, _ = curve_fit(
                    model_cont_correction_eboss,
                    xdata=k,
                    ydata=ratio,
                    sigma=err_fit,
                    p0=[1.0, 0.0, 0.0],
                    bounds=(
                        [-np.inf, -5, 0],
                        [np.inf, 5, 5],
                    ),
                )
                k_th = np.linspace(np.min(k), np.max(k), 2000)
                axplt.plot(
                    k_th,
                    model_cont_correction_eboss(k_th, *popt),
                    color=colors[i],
                    ls="--",
                )
                params.append(popt)

            elif model == "rogers":
                func = partial(hcd.rogers, z)
                if pk.velunits:
                    b0_rogers = 429.58
                else:
                    b0_rogers = 7.316
                popt, _ = curve_fit(
                    func,
                    xdata=k,
                    ydata=ratio,
                    sigma=err_fit,
                    p0=[0.8633, 0.2943, b0_rogers, -0.4964, 1.0, 0.01],
                    bounds=(
                        [0, 0, 0, -np.inf, 0, 0],
                        [np.inf, np.inf, np.inf, 0, np.inf, np.inf],
                    ),
                )
                k_th = np.linspace(np.min(k), np.max(k), 2000)
                axplt.plot(k_th, func(k_th, *popt), color=f"C{i}")
                params.append(popt)

            elif model == "skylines":
                popt, _ = curve_fit(
                    model_skylines_desi,
                    xdata=k,
                    ydata=ratio,
                    sigma=err_fit,
                    p0=[1.0, 1.0, 1.0, 1.0],
                    bounds=(
                        [-np.inf, -np.inf, -np.inf, -np.inf],
                        [np.inf, np.inf, np.inf, np.inf],
                    ),
                )
                k_th = np.linspace(np.min(k), np.max(k), 2000)
                axplt.plot(
                    k_th,
                    model_skylines_desi(k_th, *popt),
                    color=colors[i],
                    ls="--",
                )
                params.append(popt)
            elif model == "resolution_desi":
                popt, _ = curve_fit(
                    model_resolution_desi,
                    xdata=k,
                    ydata=ratio,
                    sigma=err_fit,
                    p0=[1.0, 1.0],
                    bounds=(
                        [-np.inf, -np.inf],
                        [np.inf, np.inf],
                    ),
                )
                k_th = np.linspace(np.min(k), np.max(k), 2000)
                axplt.plot(
                    k_th,
                    model_resolution_desi(k_th, *popt),
                    color=colors[i],
                    ls="--",
                )
                params.append(popt)

    z_arr, k_arr, ratio_arr, err_arr = (
        np.concatenate(z_arr),
        np.concatenate(k_arr),
        np.concatenate(ratio_arr),
        np.concatenate(err_arr),
    )

    if pk.velunits:
        ax[1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize
        )
    else:
        ax[1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize)
    ax[0].set_ylabel(ylabel, fontsize=fontsize_y)
    ax[1].set_ylabel(ylabel, fontsize=fontsize_y)
    ax[0].xaxis.set_tick_params(labelsize=fontticks)
    ax[0].yaxis.set_tick_params(labelsize=fontticks)
    ax[1].xaxis.set_tick_params(labelsize=fontticks)
    ax[1].yaxis.set_tick_params(labelsize=fontticks)
    ax[0].legend(ncol=2, fontsize=fontlegend)
    ax[1].legend(ncol=2, fontsize=fontlegend)
    ax[0].set_ylim(ymin, ymax)
    ax[1].set_ylim(ymin, ymax)
    ax[0].margins(x=0)
    fig.tight_layout()

    if pk.velunits:
        fig.savefig(os.path.join(path_out, f"{name_correction}_{suffix}_kms.pdf"))
        fig.savefig(os.path.join(path_out, f"{name_correction}_{suffix}_kms.png"))
        pickle.dump(
            params,
            open(
                os.path.join(path_out, f"{name_correction}_{suffix}_kms.pickle"), "wb"
            ),
        )
        np.savetxt(
            os.path.join(path_out, f"{name_correction}_{suffix}_kms.txt"),
            np.transpose(np.stack([z_arr, k_arr, ratio_arr, err_arr])),
            header="REDSHIFT & WAVENUMBER [s.km^-1] & RATIO POWER SPECTRA & ERROR RATIO",
        )
    else:
        fig.savefig(os.path.join(path_out, f"{name_correction}_{suffix}.pdf"))
        fig.savefig(os.path.join(path_out, f"{name_correction}_{suffix}.png"))
        pickle.dump(
            params,
            open(os.path.join(path_out, f"{name_correction}_{suffix}.pickle"), "wb"),
        )
        np.savetxt(
            os.path.join(path_out, f"{name_correction}_{suffix}.txt"),
            np.transpose(np.stack([z_arr, k_arr, ratio_arr, err_arr])),
            header="REDSHIFT & WAVENUMBER [Ang^-1] & RATIO POWER SPECTRA & ERROR RATIO",
        )
    return params


def plot_and_compute_average_ratio_power_resolution(
    pk,
    pk2,
    path_out,
    name_correction,
    suffix,
    zmax,
    kmin_AA,
    kmax_AA,
    derive_velcor=False,
    zmax_velcor=None,
    **plt_args,
):
    """Plot and fit the redshift-averaged ratio of two power spectra (resolution correction).

    For each redshift bin below ``zmax``, computes the ratio of the
    normalized power spectra ``pk.norm_p / pk2.norm_p`` (e.g. RAW over CCD
    resolution-degraded P1D) over the k-range ``[kmin_AA, kmax_AA]``
    (converted to s/km if ``pk.velunits`` is True), averages the ratio
    across redshift bins, fits it with :func:`model_resolution_desi`, plots
    the individual and averaged ratios with the best-fit curve, and saves
    the figure, fit parameters (pickle) and averaged data (txt) to
    ``path_out``.

    Args:
        pk: Power spectrum object with attributes ``zbin``, ``k``, ``norm_p``,
            ``norm_err``, ``velunits`` (numerator of the ratio).
        pk2: Power spectrum object with the same attributes as ``pk`` (denominator
            of the ratio).
        path_out (str): Output directory for the figure, pickle and txt files.
        name_correction (str): Name used as prefix of the saved file names.
        suffix (str): Suffix used in the saved file names.
        zmax (float): Maximum redshift (exclusive) of the bins to include.
        kmin_AA (float): Minimum wavenumber in Angstrom^-1 used to mask the data.
        kmax_AA (float): Maximum wavenumber in Angstrom^-1 used to mask the data.
        derive_velcor (bool, optional): If True, also derive per-redshift
            fit parameters converted to velocity units (s/km) for the
            redshift bins below ``zmax_velcor``. Defaults to False.
        zmax_velcor (float, optional): Maximum redshift (exclusive) used when
            ``derive_velcor`` is True. Required in that case.
        **plt_args: Additional plotting options, including:
            labelsize: tick label size.
            fontsize_x: x-axis label font size.
            fontsize_y: y-axis label font size.
            ymin: minimum y value for the plot.
            ymax: maximum y value for the plot.

    Returns:
        None
    """
    k_tot, ratio_tot, err_tot = [], [], []
    labelsize = utils.return_key(plt_args, "labelsize", 18)
    fontsize_x = utils.return_key(plt_args, "fontsize_x", 21)
    fontsize_y = utils.return_key(plt_args, "fontsize_y", 22)
    ymin = utils.return_key(plt_args, "ymin", 0.9)
    ymax = utils.return_key(plt_args, "ymax", 1.02)

    plt.figure(figsize=(8, 5))

    zbins = pk.zbin[pk.zbin < zmax]

    for z in zbins:
        if pk.velunits:
            kmax = float(utils.kAAtokskm(kmax_AA, z=z))
            kmin = float(utils.kAAtokskm(kmin_AA, z=z))
        else:
            kmax = kmax_AA
            kmin = kmin_AA

        k = pk.k[z]
        ratio = pk.norm_p[z] / pk2.norm_p[z]
        err_ratio = (pk.norm_p[z] / pk2.norm_p[z]) * np.sqrt(
            (pk.norm_err[z] / pk.norm_p[z]) ** 2
            + (pk2.norm_err[z] / pk2.norm_p[z]) ** 2
        )
        mask = (
            np.isnan(k)
            | np.isnan(ratio)
            | np.isnan(err_ratio)
            | (k > kmax)
            | (k < kmin)
        )

        k[mask] = np.nan
        ratio[mask] = np.nan
        err_ratio[mask] = np.nan

        plt.plot(k, ratio, alpha=0.5, ls=":", color=f"k")

        k_tot.append(k)
        ratio_tot.append(ratio)
        err_tot.append(err_ratio)

    k_tot = np.nanmean(k_tot, axis=0)
    ratio_tot = np.nanmean(ratio_tot, axis=0)
    err_tot = np.nanmean(err_tot, axis=0)
    mask = np.isnan(k_tot) | np.isnan(ratio_tot) | np.isnan(err_tot)
    k_tot = k_tot[~mask]
    ratio_tot = ratio_tot[~mask]
    err_tot = err_tot[~mask]

    plt.errorbar(
        k_tot, ratio_tot, err_tot, marker=".", markersize=10, ls="None", color="C0"
    )

    param = curve_fit(
        model_resolution_desi, k_tot, ratio_tot, sigma=err_tot, p0=[1.0, 1.0]
    )[0]
    k = np.linspace(np.min(k_tot), np.max(k_tot), 1000)
    if derive_velcor:
        param_kms = []
        zbins_velcor = pk.zbin[pk.zbin < zmax_velcor]
        for i, z in enumerate(zbins_velcor):
            param_kms.append(
                [
                    param[0],
                    param[1],
                    param[2] * utils.return_conversion_factor(z),
                ]
            )
    plt.plot(k, model_resolution_desi(k, *param))

    plt.ylabel(
        r"$\langle P_{1\mathrm{D},\alpha,\mathrm{RAW}} / P_{1\mathrm{D},\alpha,\mathrm{CCD}} \rangle_z$",
        fontsize=fontsize_y,
    )
    if pk.velunits:
        plt.xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
    else:
        plt.xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
    plt.ylim([ymin, ymax])

    plt.gca().xaxis.set_tick_params(labelsize=labelsize)
    plt.gca().yaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()

    if pk.velunits:
        plt.savefig(os.path.join(path_out, f"{name_correction}_{suffix}_kms.pdf"))
        plt.savefig(os.path.join(path_out, f"{name_correction}_{suffix}_kms.png"))
        pickle.dump(
            param,
            open(
                os.path.join(path_out, f"{name_correction}_{suffix}_kms.pickle"), "wb"
            ),
        )
        np.savetxt(
            os.path.join(path_out, f"{name_correction}_{suffix}_kms.txt"),
            np.transpose(np.stack([k_tot, ratio_tot, err_tot])),
            header="WAVENUMBER [s.km^-1] & MEAN POWER SPECTRUM RATIO & ERROR RATIO",
        )
    else:
        plt.savefig(os.path.join(path_out, f"{name_correction}_{suffix}.pdf"))
        plt.savefig(os.path.join(path_out, f"{name_correction}_{suffix}.png"))
        pickle.dump(
            param,
            open(os.path.join(path_out, f"{name_correction}_{suffix}.pickle"), "wb"),
        )
        if derive_velcor:
            pickle.dump(
                param_kms,
                open(
                    os.path.join(path_out, f"{name_correction}_{suffix}_kms.pickle"),
                    "wb",
                ),
            )
        np.savetxt(
            os.path.join(path_out, f"{name_correction}_{suffix}.txt"),
            np.transpose(np.stack([k_tot, ratio_tot, err_tot])),
            header="WAVENUMBER [Ang^-1] & MEAN POWER SPECTRUM RATIO & ERROR RATIO",
        )


###################################################
############## Applying corrections ###############
###################################################


######### Corrections masking + continuum #########


def model_skylines_desi(k, a, b, c, d):
    """Rational function model used to fit the DESI sky-line masking correction.

    Args:
        k (array_like): Wavenumber, in Angstrom^-1 or s/km.
        a (float): Numerator constant term.
        b (float): Numerator linear coefficient.
        c (float): Denominator constant term.
        d (float): Denominator linear coefficient.

    Returns:
        array_like: Correction ratio ``(a + b * k) / (c + d * k)``.
    """
    return (a + b * k) / (c + d * k)


def model_resolution_desi(k, a, b):
    """Gaussian-type model used to fit the DESI CCD resolution correction.

    Args:
        k (array_like): Wavenumber, in Angstrom^-1 or s/km.
        a (float): Amplitude.
        b (float): Damping (Gaussian) coefficient.

    Returns:
        array_like: Correction ratio ``a * exp(-b * k**2)``.
    """
    return a * np.exp(-b * k**2)


def load_model_resolution_desi(a, b):
    """Build a resolution-correction callable with fixed best-fit parameters.

    Args:
        a (float): Amplitude parameter passed to :func:`model_resolution_desi`.
        b (float): Damping coefficient passed to :func:`model_resolution_desi`.

    Returns:
        callable: Function ``func(k)`` evaluating
        :func:`model_resolution_desi` at wavenumber ``k`` with ``a`` and ``b``
        fixed.
    """
    def func(k):
        """Evaluate the resolution correction model at fixed (a, b).

        Args:
            k (array_like): Wavenumber, in Angstrom^-1 or s/km.

        Returns:
            array_like: Correction ratio at ``k``.
        """
        return model_resolution_desi(k, a, b)

    return func


def prepare_bal_correction(zbins, file_correction_bal):
    """Load per-redshift-bin BAL masking correction polynomials.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        file_correction_bal (str): Path to a pickle file containing a list
            of polynomial coefficient arrays (one per redshift bin, in
            ``numpy.poly1d`` order), as produced by
            :func:`plot_and_compute_ratio_power` with ``model="poly"``.

    Returns:
        dict: Mapping from redshift bin value to a ``numpy.poly1d`` giving
        the BAL correction as a function of ``k``. Redshift bins without a
        stored correction get the identity polynomial ``poly1d(1)``.
    """
    param_bal = pickle.load(open(file_correction_bal, "rb"))
    A_bal = {}
    for iz, z in enumerate(zbins):
        if iz >= len(param_bal):
            print(f"Redshift bin {z} have no bal correction")
            A_bal[z] = np.poly1d(1)
        else:
            A_bal[z] = np.poly1d(param_bal[iz])
    return A_bal


def prepare_hcd_correction(zbins, file_correction_hcd):
    """Load per-redshift-bin HCD (high column density absorber) correction polynomials.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        file_correction_hcd (str): Path to a pickle file containing a list
            of polynomial coefficient arrays (one per redshift bin, in
            ``numpy.poly1d`` order).

    Returns:
        dict: Mapping from redshift bin value to a ``numpy.poly1d`` giving
        the HCD correction as a function of ``k``. Redshift bins without a
        stored correction get the identity polynomial ``poly1d(1)``.
    """
    param_hcd = pickle.load(open(file_correction_hcd, "rb"))
    A_hcd = {}
    for iz, z in enumerate(zbins):
        if iz >= len(param_hcd):
            print(f"Redshift bin {z} have no hcd correction")
            A_hcd[z] = np.poly1d(1)
        else:
            A_hcd[z] = np.poly1d(param_hcd[iz])
    return A_hcd


def prepare_lines_correction(zbins, file_correction_lines):
    """Load per-redshift-bin sky-line masking correction polynomials.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        file_correction_lines (str): Path to a pickle file containing a
            list of polynomial coefficient arrays (one per redshift bin, in
            ``numpy.poly1d`` order).

    Returns:
        dict: Mapping from redshift bin value to a ``numpy.poly1d`` giving
        the line-masking correction as a function of ``k``. Redshift bins
        without a stored correction get the identity polynomial
        ``poly1d(1)``.
    """
    param_lines = pickle.load(open(file_correction_lines, "rb"))
    A_lines = {}
    for iz, z in enumerate(zbins):
        if iz >= len(param_lines):
            print(f"Redshift bin {z} have no line correction")
            A_lines[z] = np.poly1d(1)
        else:
            A_lines[z] = np.poly1d(param_lines[iz])
    return A_lines


def prepare_cont_correction(zbins, file_correction_cont):
    """Load per-redshift-bin continuum-fitting correction polynomials.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        file_correction_cont (str): Path to a pickle file containing a
            list of polynomial coefficient arrays (one per redshift bin, in
            ``numpy.poly1d`` order).

    Returns:
        dict: Mapping from redshift bin value to a ``numpy.poly1d`` giving
        the continuum correction as a function of ``k``. Redshift bins
        without a stored correction get the identity polynomial
        ``poly1d(1)``.
    """
    param_cont = pickle.load(open(file_correction_cont, "rb"))
    A_cont = {}
    for iz, z in enumerate(zbins):
        if iz >= len(param_cont):
            print(f"Redshift bin {z} have no continuum correction")
            A_cont[z] = np.poly1d(1)
        else:
            A_cont[z] = np.poly1d(param_cont[iz])
    return A_cont


def prepare_resolution_correction(zbins, file_correction_resolution):
    """Load the CCD resolution correction model for each redshift bin.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        file_correction_resolution (str): Path to a pickle file containing
            either a single ``(a, b)`` parameter pair (applied to every
            redshift bin) or a list of ``(a, b)`` pairs, one per redshift
            bin, as produced by
            :func:`plot_and_compute_average_ratio_power_resolution`.

    Returns:
        dict: Mapping from redshift bin value to a callable (from
        :func:`load_model_resolution_desi`) giving the resolution
        correction as a function of ``k``. Redshift bins without a stored
        correction get the identity polynomial ``poly1d(1)``.
    """
    param_res = pickle.load(open(file_correction_resolution, "rb"))
    A_resolution = {}
    if np.array(param_res).ndim == 1:
        for _, z in enumerate(zbins):
            A_resolution[z] = load_model_resolution_desi(*param_res)
    else:
        for iz, z in enumerate(zbins):
            if iz >= len(param_res):
                print(f"Redshift bin {z} have no resoltution correction")
                A_resolution[z] = np.poly1d(1)
            else:
                A_resolution[z] = load_model_resolution_desi(*param_res[iz])
    return A_resolution


def apply_correction(pk, zmax, file_correction, type_correction):
    """Apply a DESI-derived multiplicative correction to a power spectrum, in place.

    Dispatches to ``prepare_{type_correction}_correction`` (e.g.
    :func:`prepare_hcd_correction`, :func:`prepare_lines_correction`,
    :func:`prepare_cont_correction`, :func:`prepare_resolution_correction`,
    :func:`prepare_bal_correction`) to build the per-redshift-bin correction
    model, then multiplies ``pk.p``, ``pk.norm_p``, ``pk.err`` and
    ``pk.norm_err`` by the correction evaluated at ``pk.k[z]`` for every
    redshift bin below ``zmax``.

    Args:
        pk: Power spectrum object with attributes ``zbin``, ``k``, ``p``,
            ``norm_p``, ``err``, ``norm_err``, modified in place.
        zmax (float): Maximum redshift (exclusive) of the bins to correct.
        file_correction (str): Path to the pickle file holding the
            correction parameters, passed to the corresponding
            ``prepare_*_correction`` function.
        type_correction (str): Correction type, one of ``"hcd"``,
            ``"lines"``, ``"cont"``, ``"resolution"``, ``"bal"`` (must match
            an existing ``prepare_{type_correction}_correction`` function).

    Returns:
        None
    """
    zbins = pk.zbin[pk.zbin < zmax]
    A_corr = eval(f"prepare_{type_correction}_correction")(zbins, file_correction)
    for z in zbins:
        pk.p[z] = pk.p[z] * A_corr[z](pk.k[z])
        pk.norm_p[z] = pk.norm_p[z] * A_corr[z](pk.k[z])
        pk.err[z] = pk.err[z] * A_corr[z](pk.k[z])
        pk.norm_err[z] = pk.norm_err[z] * A_corr[z](pk.k[z])


def prepare_hcd_correction_eboss(zbins, hcd_eboss_name):
    """Load per-redshift-bin eBOSS HCD correction polynomials from an ASCII table.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        hcd_eboss_name (str): Path to a whitespace-separated text file with
            one row of ``numpy.poly1d``-ordered polynomial coefficients per
            redshift bin, readable by ``numpy.loadtxt``.

    Returns:
        dict: Mapping from redshift bin value to a ``numpy.poly1d`` giving
        the eBOSS HCD correction as a function of ``k``.
    """
    param_hcd_eboss = np.loadtxt(hcd_eboss_name)
    A_hcd = {}
    for iz, z in enumerate(zbins):
        A_hcd[z] = np.poly1d(param_hcd_eboss[iz])
    return A_hcd


def prepare_lines_correction_eboss(zbins, lines_eboss_name):
    """Load per-redshift-bin eBOSS sky-line correction polynomials from an ASCII table.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        lines_eboss_name (str): Path to a whitespace-separated text file
            with one row of ``numpy.poly1d``-ordered polynomial
            coefficients per redshift bin, readable by ``numpy.loadtxt``.

    Returns:
        dict: Mapping from redshift bin value to a ``numpy.poly1d`` giving
        the eBOSS line-masking correction as a function of ``k``.
    """
    param_lines_eboss = np.loadtxt(lines_eboss_name)
    A_lines = {}
    for iz, z in enumerate(zbins):
        A_lines[z] = np.poly1d(param_lines_eboss[iz])
    return A_lines


def model_cont_correction_eboss(a0, a1, k):
    """eBOSS continuum-fitting correction model (inverse-k plus offset).

    Args:
        a0 (float): Coefficient of the ``1/k`` term.
        a1 (float): Constant offset term.
        k (array_like): Wavenumber, in Angstrom^-1 or s/km.

    Returns:
        array_like: Correction value ``(a0 / k) + a1``.
    """
    return (a0 / k) + a1


def prepare_cont_correction_eboss(zbins, cont_eboss_name):
    """Load per-redshift-bin eBOSS continuum correction models from an ASCII table.

    For the first 6 redshift bins the correction uses the
    :func:`model_cont_correction_eboss` functional form (fit with
    parameters ``a0``, ``a1``); for the remaining bins it uses a
    ``numpy.poly1d`` built from the row of stored coefficients.

    Args:
        zbins (array_like): Redshift bin centers to build corrections for.
        cont_eboss_name (str): Path to a whitespace-separated text file
            with one row of coefficients per redshift bin, readable by
            ``numpy.loadtxt``.

    Returns:
        dict: Mapping from redshift bin value to a callable giving the
        eBOSS continuum correction as a function of ``k``.
    """
    param_cont_eboss = np.loadtxt(cont_eboss_name)
    A_cont = {}
    for iz, z in enumerate(zbins):
        if iz < 6:
            A_cont[z] = partial(
                model_cont_correction_eboss,
                param_cont_eboss[iz][0],
                param_cont_eboss[iz][1],
            )
        else:
            A_cont[z] = np.poly1d(param_cont_eboss[iz])
    return A_cont


def apply_correction_eboss(pk, zmax, file_correction, type_correction):
    """Apply an eBOSS-derived multiplicative correction to a power spectrum, in place.

    Dispatches to ``prepare_{type_correction}_correction_eboss`` (e.g.
    :func:`prepare_hcd_correction_eboss`,
    :func:`prepare_lines_correction_eboss`,
    :func:`prepare_cont_correction_eboss`) to build the per-redshift-bin
    correction model, then divides ``pk.p``, ``pk.norm_p``, ``pk.err`` and
    ``pk.norm_err`` by the correction evaluated at ``pk.k[z]`` for every
    redshift bin below ``zmax``.

    Args:
        pk: Power spectrum object with attributes ``zbin``, ``k``, ``p``,
            ``norm_p``, ``err``, ``norm_err``, modified in place.
        zmax (float): Maximum redshift (exclusive) of the bins to correct.
        file_correction (str): Path to the ASCII file holding the
            correction parameters, passed to the corresponding
            ``prepare_*_correction_eboss`` function.
        type_correction (str): Correction type, one of ``"hcd"``,
            ``"lines"``, ``"cont"`` (must match an existing
            ``prepare_{type_correction}_correction_eboss`` function).

    Returns:
        None
    """
    zbins = pk.zbin[pk.zbin < zmax]
    A_corr = eval(f"prepare_{type_correction}_correction_eboss")(zbins, file_correction)
    for z in zbins:
        pk.p[z] = pk.p[z] / A_corr[z](pk.k[z])
        pk.norm_p[z] = pk.norm_p[z] / A_corr[z](pk.k[z])
        pk.err[z] = pk.err[z] / A_corr[z](pk.k[z])
        pk.norm_err[z] = pk.norm_err[z] / A_corr[z](pk.k[z])


############## Side band corrections ##############


def prepare_metal_subtraction(zbins, file_metal, velunits=False):
    """Build the per-redshift-bin metal side-band power spectrum model.

    If ``file_metal`` is a path to a pickle file, it is expected to contain
    a ``(param_SB1_mean, param_SB1_indiv)`` tuple, and the metal power
    spectrum model for each redshift bin is built from
    :func:`p1desi.metals.model_SB1_indiv_kms` (if ``velunits`` is True) or
    :func:`p1desi.metals.model_SB1_indiv` (otherwise), bound to the mean
    parameters and the individual redshift-bin parameters. Otherwise,
    ``file_metal`` is treated as a power spectrum object (with ``k`` and
    ``p`` attributes indexed by redshift) and the model is a linear
    interpolation of its measured power spectrum.

    Args:
        zbins (array_like): Redshift bin centers to build the model for.
        file_metal (str or object): Path to a pickle file with SB1 fit
            parameters, or a power-spectrum-like object with ``k`` and
            ``p`` dict attributes keyed by redshift.
        velunits (bool, optional): If True, use the velocity-unit (s/km)
            metal model instead of the wavelength-unit (Angstrom^-1) one.
            Only used when ``file_metal`` is a pickle path. Defaults to
            False.

    Returns:
        dict: Mapping from redshift bin value to a callable giving the
        metal side-band power spectrum as a function of ``k``.
    """
    P_metal_m = {}
    if type(file_metal) == str:
        (param_SB1_mean, param_SB1_indiv) = pickle.load(open(file_metal, "rb"))
        for iz, z in enumerate(zbins):
            if velunits:
                P_metal_m[z] = partial(
                    metals.model_SB1_indiv_kms,
                    param_SB1_mean,
                    param_SB1_indiv[iz][0],
                    param_SB1_indiv[iz][1],
                )
            else:
                P_metal_m[z] = partial(
                    metals.model_SB1_indiv,
                    param_SB1_mean,
                    z,
                    param_SB1_indiv[iz][0],
                    param_SB1_indiv[iz][1],
                )
    else:
        for iz, z in enumerate(zbins):
            P_metal_m[z] = interp1d(
                file_metal.k[z],
                file_metal.p[z],
                bounds_error=False,
                fill_value="extrapolate",
            )
    return P_metal_m


def subtract_metal(pk, zmax, file_metal):
    """Subtract the metal side-band contamination from a power spectrum, in place.

    Builds the per-redshift-bin metal model via
    :func:`prepare_metal_subtraction` and subtracts it from ``pk.p``
    (directly) and from ``pk.norm_p`` (weighted by ``k / pi``, matching the
    usual P1D normalization) for every redshift bin below ``zmax``.

    Args:
        pk: Power spectrum object with attributes ``zbin``, ``k``, ``p``,
            ``norm_p``, ``velunits``, modified in place.
        zmax (float): Maximum redshift (exclusive) of the bins to correct.
        file_metal (str or object): Metal correction source, forwarded to
            :func:`prepare_metal_subtraction`.

    Returns:
        None
    """
    zbins = pk.zbin[pk.zbin < zmax]
    P_metal_m = prepare_metal_subtraction(zbins, file_metal, velunits=pk.velunits)
    for z in zbins:
        pk.p[z] = pk.p[z] - P_metal_m[z](pk.k[z])
        pk.norm_p[z] = pk.norm_p[z] - (pk.k[z] * P_metal_m[z](pk.k[z]) / np.pi)


def prepare_metal_correction_eboss(file_metal_eboss):
    """Load the raw eBOSS metal side-band correction table.

    Args:
        file_metal_eboss (str): Path to a whitespace-separated text file
            with columns ``[redshift, wavenumber, side-band power]``,
            readable by ``numpy.loadtxt``.

    Returns:
        numpy.ndarray: Array of shape ``(n_rows, 3)`` with columns
        ``[z, k, sideband]``.
    """
    param_sb_eboss = np.loadtxt(file_metal_eboss)
    return param_sb_eboss


def subtract_metal_eboss(pk, zmax, file_metal_eboss):
    """Subtract the eBOSS metal side-band correction from a power spectrum, in place.

    Loads the raw eBOSS side-band table via
    :func:`prepare_metal_correction_eboss`, and for each redshift bin below
    ``zmax`` linearly interpolates/extrapolates the tabulated side-band
    power (defined on a fixed eBOSS k-grid of ``nbin_eboss`` bins between
    ``klimInf_eboss`` and ``klimSup_eboss``) onto the bin centers of that
    grid, then subtracts it from ``pk.p`` (directly) and from ``pk.norm_p``
    (weighted by ``k / pi``).

    Args:
        pk: Power spectrum object with attributes ``zbin``, ``k``, ``p``,
            ``norm_p``, modified in place.
        zmax (float): Maximum redshift (exclusive) of the bins to correct.
        file_metal_eboss (str): Path to the eBOSS side-band ASCII table,
            forwarded to :func:`prepare_metal_correction_eboss`.

    Returns:
        None
    """
    zbins = pk.zbin[pk.zbin < zmax]
    param_sb_eboss = prepare_metal_correction_eboss(file_metal_eboss)
    nbin_eboss = 35
    klimInf_eboss = 0.000813
    klimSup_eboss = klimInf_eboss + nbin_eboss * 0.000542

    for z in zbins:
        mask_sb = param_sb_eboss[:, 0] == z
        k_sb_eboss = param_sb_eboss[:, 1][mask_sb]
        sb_eboss = param_sb_eboss[:, 2][mask_sb]

        deltak_eboss = (klimSup_eboss - klimInf_eboss) / nbin_eboss
        ib_eboss = ((k_sb_eboss - klimInf_eboss) / deltak_eboss).astype(int)
        cor_sb = np.zeros(k_sb_eboss.shape)
        slope_eboss = np.zeros(k_sb_eboss.shape)

        mask_k_sb_1 = k_sb_eboss < klimInf_eboss
        slope_eboss[mask_k_sb_1] = (sb_eboss[1] - sb_eboss[0]) / deltak_eboss
        cor_sb[mask_k_sb_1] = sb_eboss[0] + slope_eboss[mask_k_sb_1] * (
            k_sb_eboss[mask_k_sb_1] - deltak_eboss / 2 - klimInf_eboss
        )

        mask_k_sb_2 = (ib_eboss < nbin_eboss - 1) & (~mask_k_sb_1)
        slope_eboss[mask_k_sb_2] = (
            sb_eboss[ib_eboss[mask_k_sb_2] + 1] - sb_eboss[ib_eboss[mask_k_sb_2]]
        ) / deltak_eboss
        cor_sb[mask_k_sb_2] = sb_eboss[ib_eboss[mask_k_sb_2]] + slope_eboss[
            mask_k_sb_2
        ] * (
            k_sb_eboss[mask_k_sb_2]
            - deltak_eboss / 2
            - (klimInf_eboss + ib_eboss[mask_k_sb_2] * deltak_eboss)
        )

        mask_k_sb_3 = (~mask_k_sb_2) & (~mask_k_sb_1)
        slope_eboss[mask_k_sb_3] = (sb_eboss[-1] - sb_eboss[-2]) / deltak_eboss
        cor_sb[mask_k_sb_3] = sb_eboss[-1] + slope_eboss[mask_k_sb_3] * (
            k_sb_eboss[mask_k_sb_3] - deltak_eboss / 2 - (klimSup_eboss - deltak_eboss)
        )

        pk.p[z] = pk.p[z] - cor_sb
        pk.norm_p[z] = pk.norm_p[z] - (pk.k[z] * cor_sb / np.pi)


################ Noise corrections ################


def model_noise_correction(SNR, A, g, B):
    """Power-law model for the missing pipeline noise power as a function of forest SNR.

    Args:
        SNR (array_like or float): Per-forest mean signal-to-noise ratio
            (``MEANSNR``).
        A (float): Power-law amplitude.
        g (float): Power-law index (exponent applied to ``SNR ** (-g)``).
        B (float): Constant offset (asymptotic noise power at high SNR).

    Returns:
        array_like or float: Missing noise power, ``A * SNR ** (-g) + B``.
    """
    power_law = A * SNR ** (-g) + B
    return power_law


################ Apply corrections ################


def correct_individual_pk_noise(pk_in, pk_out, qsocat, correction):
    """Apply a per-survey missing-noise-power correction to individual picca Pk1D files.

    Reads every ``Pk1D-*`` FITS file in ``pk_in``; for each per-forest HDU,
    looks up the line-of-sight's survey from ``qsocat`` (matching
    ``LOS_ID``/``TARGETID``), computes the missing noise power with
    :func:`model_noise_correction` using the forest's ``MEANSNR`` header
    keyword and the survey-specific ``correction`` parameters, adds it to
    the existing ``PK_NOISE`` column, stores it in a new ``PK_NOISE_MISS``
    column, and writes the corrected HDUs to a new file of the same name in
    ``pk_out``.

    Args:
        pk_in (str): Directory containing the input ``Pk1D-*`` FITS files.
        pk_out (str): Output directory for the corrected FITS files
            (created if it does not exist).
        qsocat (str): Path to the QSO catalog FITS file (``QSO_CAT`` HDU
            with ``TARGETID`` and ``SURVEY`` columns) used to map each
            line of sight to its survey.
        correction (dict): Mapping from survey name to the
            ``(A, g, B)`` parameter tuple passed to
            :func:`model_noise_correction`.

    Returns:
        None
    """
    os.makedirs(pk_out, exist_ok=True)
    pk_files = glob.glob(os.path.join(pk_in, "Pk1D-*"))

    qso_file = fitsio.FITS(qsocat)["QSO_CAT"]
    targetid_qso = qso_file["TARGETID"][:]
    survey_qso = qso_file["SURVEY"][:]
    for i in range(len(pk_files)):
        pk_out_name = pk_files[i].split("/")[-1]
        f_out = fitsio.FITS(os.path.join(pk_out, pk_out_name), "rw", clobber=True)
        f = fitsio.FITS(pk_files[i])
        for j in range(1, len(f)):
            header = dict(f[j].read_header())
            snr = header["MEANSNR"]
            id = header["LOS_ID"]
            survey_id = survey_qso[np.argwhere(targetid_qso == id)[0]][0]
            p_noise_miss = model_noise_correction(snr, *correction[survey_id])
            line = f[j].read()
            new_line = np.zeros(
                line.size,
                dtype=[
                    (line.dtype.names[i], line.dtype[i]) for i in range(len(line.dtype))
                ]
                + [("PK_NOISE_MISS", ">f8")],
            )
            for name in line.dtype.names:
                new_line[name] = line[name]
            new_line["PK_NOISE"] = new_line["PK_NOISE"] + p_noise_miss
            new_line["PK_NOISE_MISS"] = np.full(line.size, p_noise_miss)
            f_out.write(new_line, header=header, extname=str(id))


def apply_p1d_corections(
    pk,
    zmax,
    apply_DESI_maskcont_corr,
    apply_eBOSS_maskcont_corr,
    apply_DESI_sb_corr,
    apply_eBOSS_sb_corr,
    correction_to_apply,
    file_correction_hcd=None,
    file_correction_hcd_eboss=None,
    file_correction_lines=None,
    file_correction_lines_eboss=None,
    file_correction_cont=None,
    file_correction_cont_eboss=None,
    file_correction_resolution=None,
    file_correction_bal=None,
    file_metal=None,
    file_metal_eboss=None,
):
    """Orchestrate application of the masking/continuum and side-band corrections to a P1D.

    Depending on the boolean flags, applies either the DESI or the eBOSS
    masking/continuum corrections (for each correction type listed in
    ``correction_to_apply``, via :func:`apply_correction` or
    :func:`apply_correction_eboss`), and either the DESI or the eBOSS
    side-band metal correction (via :func:`subtract_metal` or
    :func:`subtract_metal_eboss`).

    Args:
        pk: Power spectrum object to correct, modified in place (see
            :func:`apply_correction` / :func:`apply_correction_eboss` /
            :func:`subtract_metal` / :func:`subtract_metal_eboss` for the
            attributes involved).
        zmax (float): Maximum redshift (exclusive) of the bins to correct.
        apply_DESI_maskcont_corr (bool): If True, apply the DESI
            masking/continuum corrections listed in ``correction_to_apply``.
        apply_eBOSS_maskcont_corr (bool): If True, apply the eBOSS
            masking/continuum corrections listed in ``correction_to_apply``.
        apply_DESI_sb_corr (bool): If True, apply the DESI metal side-band
            subtraction.
        apply_eBOSS_sb_corr (bool): If True, apply the eBOSS metal
            side-band subtraction.
        correction_to_apply (iterable of str): Correction type names (e.g.
            ``"hcd"``, ``"lines"``, ``"cont"``, ``"resolution"``, ``"bal"``)
            to apply via the masking/continuum branch.
        file_correction_hcd (str, optional): Pickle path for the DESI HCD
            correction, used when ``"hcd"`` is in ``correction_to_apply``
            and ``apply_DESI_maskcont_corr`` is True.
        file_correction_hcd_eboss (str, optional): ASCII table path for the
            eBOSS HCD correction, used when ``"hcd"`` is in
            ``correction_to_apply`` and ``apply_eBOSS_maskcont_corr`` is
            True.
        file_correction_lines (str, optional): Pickle path for the DESI
            line-masking correction.
        file_correction_lines_eboss (str, optional): ASCII table path for
            the eBOSS line-masking correction.
        file_correction_cont (str, optional): Pickle path for the DESI
            continuum correction.
        file_correction_cont_eboss (str, optional): ASCII table path for
            the eBOSS continuum correction.
        file_correction_resolution (str, optional): Pickle path for the
            DESI resolution correction.
        file_correction_bal (str, optional): Pickle path for the DESI BAL
            correction.
        file_metal (str or object, optional): Metal side-band source for
            the DESI correction, forwarded to :func:`subtract_metal`.
        file_metal_eboss (str, optional): ASCII table path for the eBOSS
            metal side-band correction, forwarded to
            :func:`subtract_metal_eboss`.

    Returns:
        The corrected ``pk`` object (same instance, modified in place).

    Raises:
        KeyError: Conceptually raised (the value is returned rather than
            raised, matching the current implementation) if both
            ``apply_DESI_maskcont_corr`` and ``apply_eBOSS_maskcont_corr``
            are True, or if both ``apply_DESI_sb_corr`` and
            ``apply_eBOSS_sb_corr`` are True.
    """
    if apply_DESI_maskcont_corr & apply_eBOSS_maskcont_corr:
        return KeyError("Same type of correction is applied two times")

    if apply_DESI_sb_corr & apply_eBOSS_sb_corr:
        return KeyError("Same type of correction is applied two times")

    if apply_DESI_maskcont_corr:
        for type_correction in correction_to_apply:
            apply_correction(
                pk, zmax, eval(f"file_correction_{type_correction}"), type_correction
            )

    if apply_eBOSS_maskcont_corr:
        for type_correction in correction_to_apply:
            apply_correction_eboss(
                pk,
                zmax,
                eval(f"file_correction_{type_correction}_eboss"),
                type_correction,
            )

    if apply_DESI_sb_corr:
        subtract_metal(pk, zmax, file_metal)

    if apply_eBOSS_sb_corr:
        subtract_metal_eboss(pk, zmax, file_metal_eboss)

    return pk
