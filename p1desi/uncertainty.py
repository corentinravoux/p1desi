"""Statistical and systematic uncertainty / covariance budget for the DESI Lyman-alpha P1D.

This module builds, stores and plots the error budget of the 1D Lyman-alpha
forest power spectrum (P1D) measured from picca outputs. It covers:

- Serialization (pickle) and reloading of per-redshift-bin systematic error
  terms (noise, resolution, resolution correction, side band / metals,
  line masking, DLA/BAL masking, continuum fitting, DLA/BAL completeness)
  for the EDR and DESI Y1 analyses.
- Plotting of statistical uncertainties (`sigma_stat`) as a function of
  wavenumber `k` (in Angstrom^-1 or s/km) and redshift bin.
- Plotting of the individual systematic error contributions and their total,
  combined in quadrature.
- Plotting of the (bootstrap or analytic) covariance / correlation matrix
  per redshift bin, optionally inflated with the systematic error terms.

Wavenumbers are expressed either in Angstrom^-1 or in velocity units
(s/km, `velunits=True`) depending on the `Pk1D`-like object passed in.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from p1desi import corrections, hcd, utils


def create_uncertainty_systematics(
    wavenumber,
    syste_noise,
    syste_reso,
    syste_resocorrection,
    syste_sb,
    syste_lines,
    syste_hcd,
    syste_continuum,
    syste_dla_completeness,
    syste_tot,
    name,
):
    """Pickle the EDR per-redshift-bin systematic uncertainty terms to disk.

    Args:
        wavenumber (dict): Mapping of redshift bin -> wavenumber array
            (Angstrom^-1 or s/km).
        syste_noise (dict): Mapping of redshift bin -> noise-estimation
            systematic error array.
        syste_reso (dict): Mapping of redshift bin -> resolution systematic
            error array.
        syste_resocorrection (dict): Mapping of redshift bin -> resolution
            correction systematic error array.
        syste_sb (dict): Mapping of redshift bin -> side band systematic
            error array.
        syste_lines (dict): Mapping of redshift bin -> line masking
            systematic error array.
        syste_hcd (dict): Mapping of redshift bin -> DLA masking systematic
            error array.
        syste_continuum (dict): Mapping of redshift bin -> continuum fitting
            systematic error array.
        syste_dla_completeness (dict): Mapping of redshift bin -> DLA
            completeness systematic error array.
        syste_tot (dict): Mapping of redshift bin -> total systematic error
            array (quadrature sum of the terms above).
        name (str): Output pickle file path.

    Returns:
        None
    """
    pickle.dump(
        (
            wavenumber,
            syste_noise,
            syste_reso,
            syste_resocorrection,
            syste_sb,
            syste_lines,
            syste_hcd,
            syste_continuum,
            syste_dla_completeness,
            syste_tot,
        ),
        open(name, "wb"),
    )


def prepare_uncertainty_systematics(
    file_systematics,
):
    """Load and repackage EDR systematic uncertainty terms from a pickle file.

    Args:
        file_systematics (str): Path to a pickle file previously written by
            `create_uncertainty_systematics`.

    Returns:
        tuple: A 4-tuple of:
            - wavenumber (dict): Redshift bin -> wavenumber array.
            - syste_tot (dict): Redshift bin -> total systematic error array.
            - list_systematics (list): List of dicts (redshift bin ->
              per-term systematic error array), one per named term, in the
              order matching `list_systematics_name`.
            - list_systematics_name (list of str): Human-readable labels for
              each entry of `list_systematics` (e.g. "Noise", "PSF",
              "Resolution", ...).
    """
    (
        wavenumber,
        syste_noise,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_continuum,
        syste_dla_completeness,
        syste_tot,
    ) = pickle.load(open(file_systematics, "rb"))
    list_systematics = [
        syste_noise,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_continuum,
        syste_dla_completeness,
    ]
    list_systematics_name = [
        "Noise",
        "PSF",
        "Resolution",
        "Side band",
        "Lines",
        "DLA",
        "Continuum",
        "DLA completeness",
    ]
    return wavenumber, syste_tot, list_systematics, list_systematics_name


def create_uncertainty_systematics_y1(
    wavenumber,
    syste_reso,
    syste_resocorrection,
    syste_sb,
    syste_lines,
    syste_hcd,
    syste_bal,
    syste_continuum,
    syste_dla_completeness,
    syste_bal_completeness,
    syste_tot,
    name,
    velunits,
    zbins,
):
    """Pickle DESI Y1 per-redshift-bin systematic uncertainty terms and export them to text.

    Same as `create_uncertainty_systematics` but for the DESI Y1 analysis,
    which adds BAL masking and BAL completeness terms and drops the noise
    term. In addition to pickling the terms, this also writes a flat,
    concatenated-over-redshift-bins `.txt` table (space-delimited, with a
    header) next to the pickle file.

    Args:
        wavenumber (dict): Mapping of redshift bin -> wavenumber array
            (Angstrom^-1 or s/km).
        syste_reso (dict): Mapping of redshift bin -> resolution systematic
            error array.
        syste_resocorrection (dict): Mapping of redshift bin -> resolution
            correction systematic error array.
        syste_sb (dict): Mapping of redshift bin -> side band / metal power
            spectrum systematic error array.
        syste_lines (dict): Mapping of redshift bin -> line masking
            systematic error array.
        syste_hcd (dict): Mapping of redshift bin -> DLA masking systematic
            error array.
        syste_bal (dict): Mapping of redshift bin -> BAL AI masking
            systematic error array.
        syste_continuum (dict): Mapping of redshift bin -> continuum fitting
            systematic error array.
        syste_dla_completeness (dict): Mapping of redshift bin -> DLA
            completeness systematic error array.
        syste_bal_completeness (dict): Mapping of redshift bin -> BAL
            completeness systematic error array.
        syste_tot (dict): Mapping of redshift bin -> total systematic error
            array (quadrature sum of the terms above).
        name (str): Output pickle file path (also used, with the
            `.pickle` suffix stripped, to derive the `.txt` output path).
        velunits (bool): If True, wavenumber is in s/km and the text header
            uses those units; otherwise Angstrom^-1 is assumed.
        zbins (iterable): Redshift bin values to iterate over, in the order
            used to concatenate the per-bin arrays into the text table.

    Returns:
        None
    """
    pickle.dump(
        (
            wavenumber,
            syste_reso,
            syste_resocorrection,
            syste_sb,
            syste_lines,
            syste_hcd,
            syste_bal,
            syste_continuum,
            syste_dla_completeness,
            syste_bal_completeness,
            syste_tot,
        ),
        open(name, "wb"),
    )

    if velunits:
        header = (
            "REDSHIFT & WAVENUMBER [s.km^-1] & SYST RESOLUTION &"
            + " SYST RESOLUTION CORRECTION & SYST SIDE BAND &"
            + " SYST LINES & SYST DLA & SYST BAL & SYST CONTINUUM &"
            + " SYST DLA COMPLETENESS & SYST BAL COMPLETENESS & TOTAL SYST"
        )
    else:
        header = (
            "REDSHIFT & WAVENUMBER [Ang^-1] & SYST RESOLUTION &"
            + " SYST RESOLUTION CORRECTION & SYST SIDE BAND &"
            + " SYST LINES & SYST DLA & SYST BAL & SYST CONTINUUM &"
            + " SYST DLA COMPLETENESS & SYST BAL COMPLETENESS & TOTAL SYST"
        )

    z_array = [z for z in zbins for i in range(len(wavenumber[z]))]
    wavenumber_array = np.concatenate([wavenumber[z] for z in zbins], axis=0)
    syste_reso_array = np.concatenate([syste_reso[z] for z in zbins], axis=0)
    syste_resocorrection_array = np.concatenate(
        [syste_resocorrection[z] for z in zbins], axis=0
    )
    syste_sb_array = np.concatenate([syste_sb[z] for z in zbins], axis=0)
    syste_lines_array = np.concatenate([syste_lines[z] for z in zbins], axis=0)
    syste_hcd_array = np.concatenate([syste_hcd[z] for z in zbins], axis=0)
    syste_bal_array = np.concatenate([syste_bal[z] for z in zbins], axis=0)
    syste_continuum_array = np.concatenate([syste_continuum[z] for z in zbins], axis=0)
    syste_dla_completeness_array = np.concatenate(
        [syste_dla_completeness[z] for z in zbins], axis=0
    )
    syste_bal_completeness_array = np.concatenate(
        [syste_bal_completeness[z] for z in zbins], axis=0
    )
    syste_tot_array = np.concatenate([syste_tot[z] for z in zbins], axis=0)

    np.savetxt(
        f"{name.split('.pickle')[0]}.txt",
        np.transpose(
            np.stack(
                [
                    z_array,
                    wavenumber_array,
                    syste_reso_array,
                    syste_resocorrection_array,
                    syste_sb_array,
                    syste_lines_array,
                    syste_hcd_array,
                    syste_bal_array,
                    syste_continuum_array,
                    syste_dla_completeness_array,
                    syste_bal_completeness_array,
                    syste_tot_array,
                ]
            )
        ),
        header=header,
    )


def prepare_uncertainty_systematics_y1(
    file_systematics,
):
    """Load and repackage DESI Y1 systematic uncertainty terms from a pickle file.

    Same as `prepare_uncertainty_systematics` but for the DESI Y1 error
    budget, which includes BAL masking and BAL completeness terms in
    addition to resolution, resolution correction, side band, lines, DLA
    masking, continuum fitting and DLA completeness.

    Args:
        file_systematics (str): Path to a pickle file previously written by
            `create_uncertainty_systematics_y1`.

    Returns:
        tuple: A 4-tuple of:
            - wavenumber (dict): Redshift bin -> wavenumber array.
            - syste_tot (dict): Redshift bin -> total systematic error array.
            - list_systematics (list): List of dicts (redshift bin ->
              per-term systematic error array), one per named term, in the
              order matching `list_systematics_name`.
            - list_systematics_name (list of str): Human-readable labels for
              each entry of `list_systematics` (e.g. "PSF", "Resolution",
              "Side band", "Lines", "DLA", "BAL", "Continuum",
              "DLA completeness", "BAL completeness").
    """
    (
        wavenumber,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_bal,
        syste_continuum,
        syste_dla_completeness,
        syste_bal_completeness,
        syste_tot,
    ) = pickle.load(open(file_systematics, "rb"))
    list_systematics = [
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_bal,
        syste_continuum,
        syste_dla_completeness,
        syste_bal_completeness,
    ]
    list_systematics_name = [
        "PSF",
        "Resolution",
        "Side band",
        "Lines",
        "DLA",
        "BAL",
        "Continuum",
        "DLA completeness",
        "BAL completeness",
    ]
    return wavenumber, syste_tot, list_systematics, list_systematics_name


def plot_stat_uncertainties(
    pk,
    zmax,
    outname=None,
    outpoints=None,
    pk_low_z=None,
    z_change=None,
    **plot_args,
):
    """Plot statistical P1D uncertainties and their relative size vs wavenumber.

    For each redshift bin below `zmax`, plots `sigma_stat(k)` (left panel)
    and `sigma_stat(k) / P1D(k)` (right panel) restricted to the
    `[kmin_AA, kmax_AA]` window (converted to s/km if `pk` is in velocity
    units). Optionally saves the figure and/or a flat text table of the
    plotted points.

    Args:
        pk: `Pk1D`-like object exposing `zbin`, `velunits`, `k`, `err` and
            `p` (all dict-like, keyed by redshift bin).
        zmax (float): Upper redshift bound (exclusive); only bins with
            `pk.zbin < zmax` are plotted.
        outname (str, optional): If given, base path (without extension) to
            save the figure as both `.pdf` and `.png`.
        outpoints (str, optional): If given, base path (without extension)
            to save the underlying plotted points as a `.txt` table.
        pk_low_z: `Pk1D`-like object to use instead of `pk` for redshift
            bins below `z_change`. Defaults to None (always use `pk`).
        z_change (float, optional): Redshift threshold below which
            `pk_low_z` is used instead of `pk`. Required if `pk_low_z` is
            given.
        **plot_args: Optional plotting overrides, e.g. `fontsize_x`,
            `fontsize_y`, `labelsize`, `fontl`, `kmin_AA`, `kmax_AA`,
            `ymin`, `ymax`, `ymin2`, `ymax2`, `figsize`, `color_map`
            (`"default"` or `"rainbow"`).

    Returns:
        None
    """
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.5)
    ymin = utils.return_key(plot_args, "ymin", 0.0018)
    ymax = utils.return_key(plot_args, "ymax", 0.07)
    ymin2 = utils.return_key(plot_args, "ymin2", 0.01)
    ymax2 = utils.return_key(plot_args, "ymax2", 0.2)
    figsize = utils.return_key(plot_args, "figsize", (16, 6))

    zbins = pk.zbin[pk.zbin < zmax]

    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        colors = [f"C{i}" for i, z in enumerate(zbins)]
    elif color_map == "rainbow":
        colors = cm.rainbow(np.linspace(0, 1, len(zbins)))

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    z_arr, k_arr, err_arr, err_on_p_arr = [], [], [], []
    for i, z in enumerate(zbins):
        pk_to_plot = pk
        if pk_low_z is not None:
            if z < z_change:
                pk_to_plot = pk_low_z
        if pk_to_plot.velunits:
            kmax = float(utils.kAAtokskm(kmax_AA, z=z))
            kmin = float(utils.kAAtokskm(kmin_AA, z=z))
        else:
            kmax = kmax_AA
            kmin = kmin_AA

        mask = (pk_to_plot.k[z] > kmin) & (pk_to_plot.k[z] < kmax)
        ax[0].semilogy(
            pk_to_plot.k[z][mask],
            pk_to_plot.err[z][mask],
            label=r"$z = ${:1.1f}".format(z),
            color=colors[i],
        )
        ax[1].semilogy(
            pk_to_plot.k[z][mask],
            pk_to_plot.err[z][mask] / pk_to_plot.p[z][mask],
            label=r"$z = ${:1.1f}".format(z),
            color=colors[i],
        )
        z_arr.append(np.full(pk_to_plot.k[z][mask].shape, z))
        k_arr.append(pk_to_plot.k[z][mask])
        err_arr.append(pk_to_plot.err[z][mask])
        err_on_p_arr.append(pk_to_plot.err[z][mask] / pk_to_plot.p[z][mask])

    z_arr = np.concatenate(z_arr, axis=0)
    k_arr = np.concatenate(k_arr, axis=0)
    err_arr = np.concatenate(err_arr, axis=0)
    err_on_p_arr = np.concatenate(err_on_p_arr, axis=0)

    if pk.velunits:
        ax[0].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
        ax[1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
    else:
        ax[0].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
        ax[1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)

    ax[0].legend(loc="upper center", ncol=2, fontsize=fontlegend)
    ax[0].set_ylabel(r"$\sigma_{\mathrm{stat}}$", fontsize=fontsize_y)
    ax[0].yaxis.set_tick_params(labelsize=labelsize)
    ax[0].xaxis.set_tick_params(labelsize=labelsize)
    ax[0].set_xlim(kmin, kmax)
    ax[0].set_ylim(ymin, ymax)

    ax[1].set_ylabel(
        r"$\sigma_{\mathrm{stat}}/P_{1\mathrm{D},\alpha}$", fontsize=fontsize_y
    )
    ax[1].yaxis.set_tick_params(labelsize=labelsize)
    ax[1].xaxis.set_tick_params(labelsize=labelsize)
    ax[1].set_xlim(kmin, kmax)
    ax[1].set_ylim(ymin2, ymax2)
    if outname is not None:
        fig.tight_layout()
        fig.savefig(f"{outname}.pdf")
        fig.savefig(f"{outname}.png")

    if outpoints is not None:
        if pk.velunits:
            header = "REDSHIFT & WAVENUMBER [s.km^-1] & P1D ERROR [km.s^-1] & P1D ERROR / P1D"
        else:
            header = (
                "REDSHIFT & WAVENUMBER [Ang^-1] & P1D ERROR [Ang] & P1D ERROR / P1D"
            )
        np.savetxt(
            f"{outpoints}.txt",
            np.transpose(np.stack([z_arr, k_arr, err_arr, err_on_p_arr])),
            header=header,
        )


def plot_syst_uncertainties(
    pk,
    pk_sb,
    outname,
    zmax,
    kmin,
    kmax,
    delta_l,
    delta_delta_l,
    resolution_coeff_fit,
    lines_coeff_fit,
    hcd_coeff_fit,
    continuum_coeff_fit,
    dla_completeness_coef,
    **plot_args,
):
    """Compute, plot and save the EDR systematic error budget vs wavenumber.

    For each redshift bin below `zmax`, computes the individual systematic
    error terms (noise estimation, resolution, resolution correction, side
    band, line masking, DLA masking, continuum fitting, DLA completeness)
    as fractions of the fitted correction models (from `corrections` and
    `hcd.rogers`) applied to `pk.p`, combines them in quadrature into a
    total, and plots both the absolute systematic error and its ratio to
    the statistical error (`pk.err`) in a grid of subplots (one row per
    term, plus a "Total" row). Saves the figure and pickles the computed
    terms via `create_uncertainty_systematics`.

    Args:
        pk: `Pk1D`-like object exposing `zbin`, `velunits`, `k`, `p`,
            `err`, `p_noise_miss`, `resocor` (all dict-like, keyed by
            redshift bin).
        pk_sb: `Pk1D`-like side-band power spectrum object exposing `err`
            (dict-like, keyed by redshift bin), used for the side-band
            systematic term.
        outname (str): Base output path (without extension) for the saved
            figure (`.pdf`) and the pickled systematics (`.pickle`); a
            `_kms` suffix is appended if `pk.velunits` is True.
        zmax (float): Upper redshift bound (exclusive); only bins with
            `pk.zbin < zmax` are used.
        kmin (float): Lower x-axis limit for the first subplot row.
        kmax (float): Upper x-axis limit for the first subplot row.
        delta_l (sequence): Per-redshift-bin resolution length uncertainty
            values used in the resolution systematic term.
        delta_delta_l (sequence): Per-redshift-bin uncertainty on
            `delta_l`, used in the resolution systematic term.
        resolution_coeff_fit: Fit coefficients passed to
            `corrections.prepare_resolution_correction`.
        lines_coeff_fit: Fit coefficients passed to
            `corrections.prepare_lines_correction`.
        hcd_coeff_fit: Fit coefficients passed to
            `corrections.prepare_hcd_correction`.
        continuum_coeff_fit: Fit coefficients passed to
            `corrections.prepare_cont_correction`.
        dla_completeness_coef (sequence): Per-redshift-bin coefficients
            passed to `hcd.rogers` for the DLA completeness correction; if
            a redshift bin index has no entry, the correction defaults to 1
            (no systematic).
        **plot_args: Optional plotting overrides, e.g. `fontsize_x`,
            `fontsize_y`, `fontlegend`, `title_shift`, `title_yshift`,
            `size`, `title_size`, `figsize`, `ylim_top`, `ylim_bottom`,
            `n_subplots`, `color_map` (`"default"` or `"rainbow"`).

    Returns:
        None
    """
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 18)
    fontlegend = utils.return_key(plot_args, "fontlegend", 14)
    title_shift = utils.return_key(plot_args, "title_shift", 1.1)
    title_yshift = utils.return_key(plot_args, "title_yshift", 1.05)
    size = utils.return_key(plot_args, "size", 12)
    title_size = utils.return_key(plot_args, "title_size", 14)
    figsize = utils.return_key(plot_args, "figsize", (11, 15))
    ylim_top = utils.return_key(plot_args, "ylim_top", None)
    ylim_bottom = utils.return_key(plot_args, "ylim_bottom", None)
    n_subplots = utils.return_key(plot_args, "n_subplots", 9)

    zbins = pk.zbin[pk.zbin < zmax]

    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        colors = [f"C{i}" for i, z in enumerate(zbins)]
    elif color_map == "rainbow":
        colors = cm.rainbow(np.linspace(0, 1, len(zbins)))

    fig, ax = plt.subplots(n_subplots, 2, figsize=figsize, sharex=True)

    A_lines = corrections.prepare_lines_correction(zbins, lines_coeff_fit)
    A_hcd = corrections.prepare_hcd_correction(zbins, hcd_coeff_fit)
    A_cont = corrections.prepare_cont_correction(zbins, continuum_coeff_fit)
    A_reso = corrections.prepare_resolution_correction(zbins, resolution_coeff_fit)

    (
        wavenumber_systematics,
        syste_noise,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_continuum,
        syste_dla_completeness,
        syste_tot,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})

    for iz, z in enumerate(zbins):
        syste_tot[z] = []

        syste_noise[z] = 0.3 * (pk.p_noise_miss[z] / pk.resocor[z])
        syste_tot[z].append(syste_noise[z] ** 2)
        ax[0][1].plot(pk.k[z], syste_noise[z] / pk.err[z], color=colors[iz])
        ax[0][0].plot(pk.k[z], syste_noise[z], color=colors[iz])
        ax[0][0].set_title(
            "Noise estimation", x=title_shift, y=title_yshift, fontsize=title_size
        )

        syste_reso[z] = 2 * pk.k[z] ** 2 * delta_l[iz] * delta_delta_l[iz] * pk.p[z]
        syste_tot[z].append(syste_reso[z] ** 2)
        ax[1][1].plot(pk.k[z], syste_reso[z] / pk.err[z], color=colors[iz])
        ax[1][0].plot(pk.k[z], syste_reso[z], color=colors[iz])
        ax[1][0].set_title(
            "Resolution", x=title_shift, y=title_yshift, fontsize=title_size
        )

        syste_resocorrection[z] = 0.3 * np.abs(A_reso[z](pk.k[z]) - 1) * pk.p[z]
        syste_tot[z].append(syste_resocorrection[z] ** 2)
        ax[2][1].plot(pk.k[z], syste_resocorrection[z] / pk.err[z], color=colors[iz])
        ax[2][0].plot(pk.k[z], syste_resocorrection[z], color=colors[iz])
        ax[2][0].set_title(
            "Resolution Correction",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_sb[z] = pk_sb.err[z]
        syste_tot[z].append(syste_sb[z] ** 2)
        ax[3][1].plot(pk.k[z], syste_sb[z] / pk.err[z], color=colors[iz])
        ax[3][0].plot(pk.k[z], syste_sb[z], color=colors[iz])
        ax[3][0].set_title(
            "Side band", x=title_shift, y=title_yshift, fontsize=title_size
        )

        syste_lines[z] = 0.3 * np.abs(A_lines[z](pk.k[z]) - 1) * pk.p[z]
        syste_tot[z].append(syste_lines[z] ** 2)
        ax[4][1].plot(
            pk.k[z],
            syste_lines[z] / pk.err[z],
            label=r"$z = ${:1.1f}".format(z),
            color=colors[iz],
        )
        ax[4][0].plot(pk.k[z], syste_lines[z], color=colors[iz])
        ax[4][0].set_title(
            "Line masking", x=title_shift, y=title_yshift, fontsize=title_size
        )

        syste_hcd[z] = 0.3 * np.abs(A_hcd[z](pk.k[z]) - 1) * pk.p[z]
        syste_tot[z].append(syste_hcd[z] ** 2)
        ax[5][1].plot(pk.k[z], syste_hcd[z] / pk.err[z], color=colors[iz])
        ax[5][0].plot(pk.k[z], syste_hcd[z], color=colors[iz])
        ax[5][0].set_title(
            "DLA masking", x=title_shift, y=title_yshift, fontsize=title_size
        )

        syste_continuum[z] = 0.3 * np.abs(A_cont[z](pk.k[z]) - 1) * pk.p[z]
        syste_tot[z].append(syste_continuum[z] ** 2)
        ax[6][1].plot(pk.k[z], syste_continuum[z] / pk.err[z], color=colors[iz])
        ax[6][0].plot(pk.k[z], syste_continuum[z], color=colors[iz])
        ax[6][0].set_title(
            "Continuum fitting", x=title_shift, y=title_yshift, fontsize=title_size
        )
        if iz >= len(dla_completeness_coef):
            print(f"Redshift bin {z} have no dla completeness info")
            A_dla_completeness = np.poly1d(1)
        else:
            A_dla_completeness = hcd.rogers(z, pk.k[z], *dla_completeness_coef[iz])
        syste_dla_completeness[z] = 0.2 * np.abs(A_dla_completeness - 1) * pk.p[z]
        syste_tot[z].append(syste_dla_completeness[z] ** 2)
        ax[7][1].plot(pk.k[z], syste_dla_completeness[z] / pk.err[z], color=colors[iz])
        ax[7][0].plot(pk.k[z], syste_dla_completeness[z], color=colors[iz])
        ax[7][0].set_title(
            "DLA completeness", x=title_shift, y=title_yshift, fontsize=title_size
        )

        syste_tot[z] = np.sqrt(np.sum(syste_tot[z], axis=0))
        ax[8][1].plot(pk.k[z], syste_tot[z] / pk.err[z], color=colors[iz])
        ax[8][0].plot(pk.k[z], syste_tot[z], color=colors[iz])
        ax[8][0].set_title("Total", x=title_shift, y=title_yshift, fontsize=title_size)
        wavenumber_systematics[z] = pk.k[z]

    ax[0][0].set_xlim(kmin, kmax)

    ax[4][1].legend(
        loc=2, bbox_to_anchor=(1.04, 1.25), borderaxespad=0.0, fontsize=fontlegend
    )

    for i in range(n_subplots):
        ax[i][1].set_ylabel(
            r"$\sigma_{\mathrm{syst}} / \sigma_{\mathrm{stat}}$", fontsize=fontsize_y
        )
        ax[i][0].set_ylabel(r"$\sigma_{\mathrm{syst}}$", fontsize=fontsize_y)
        ax[i][0].set_yscale("log")
        ax[i][0].yaxis.set_tick_params(labelsize=size)
        ax[i][1].yaxis.set_tick_params(labelsize=size)

    if ylim_top is not None:
        for i in range(len(ylim_top)):
            ax[i][1].set_ylim(top=ylim_top[i][1])
            ax[i][0].set_ylim(top=ylim_top[i][0])
    if ylim_bottom is not None:
        for i in range(len(ylim_bottom)):
            ax[i][1].set_ylim(bottom=ylim_bottom[i][1])
            ax[i][0].set_ylim(bottom=ylim_bottom[i][0])
    if pk.velunits:
        ax[-1][0].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
        ax[-1][0].xaxis.set_tick_params(labelsize=size)
        ax[-1][1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
        ax[-1][1].xaxis.set_tick_params(labelsize=size)
    else:
        ax[-1][0].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
        ax[-1][0].xaxis.set_tick_params(labelsize=size)
        ax[-1][1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
        ax[-1][1].xaxis.set_tick_params(labelsize=size)

    fig.subplots_adjust(wspace=0.2, hspace=0.45, right=0.85)
    if pk.velunits:
        plt.savefig(f"{outname}_kms.pdf")
        name_file = f"{outname}_kms.pickle"
    else:
        plt.savefig(f"{outname}.pdf")
        name_file = f"{outname}.pickle"
    create_uncertainty_systematics(
        wavenumber_systematics,
        syste_noise,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_continuum,
        syste_dla_completeness,
        syste_tot,
        name_file,
    )


def plot_syst_uncertainties_y1(
    pk,
    pk_sb,
    outname,
    zmax,
    kmin,
    kmax,
    delta_l,
    delta_delta_l,
    resolution_coeff_fit,
    lines_coeff_fit,
    hcd_coeff_fit,
    bal_coeff_fit,
    continuum_coeff_fit,
    dla_completeness_coef,
    bal_completeness_coef,
    pk_low_z=None,
    z_change=None,
    **plot_args,
):
    """Compute, plot and save the DESI Y1 systematic error budget vs wavenumber.

    Same as `plot_syst_uncertainties` but for the DESI Y1 analysis: adds a
    BAL AI masking term and a BAL completeness term (both derived the same
    way as their DLA counterparts), relabels the side-band term as "Metal
    power spectrum", and supports per-redshift-bin substitution of `pk` by
    `pk_low_z` below `z_change`. Saves the figure and pickles the computed
    terms (plus a text table) via `create_uncertainty_systematics_y1`.

    Args:
        pk: `Pk1D`-like object exposing `zbin`, `velunits`, `k`, `p`, `err`
            (dict-like, keyed by redshift bin); used for all bins unless
            overridden by `pk_low_z`.
        pk_sb: `Pk1D`-like side-band power spectrum object exposing `err`
            (dict-like, keyed by redshift bin), used for the metal power
            spectrum systematic term.
        outname (str): Base output path (without extension) for the saved
            figure (`.pdf`) and the pickled systematics (`.pickle`); a
            `_kms` suffix is appended if `pk.velunits` is True.
        zmax (float): Upper redshift bound (exclusive); only bins with
            `pk.zbin < zmax` are used.
        kmin (float): Lower x-axis limit for the first subplot row.
        kmax (float): Upper x-axis limit for the first subplot row.
        delta_l (sequence): Per-redshift-bin resolution length uncertainty
            values used in the resolution systematic term.
        delta_delta_l (sequence): Per-redshift-bin uncertainty on
            `delta_l`, used in the resolution systematic term.
        resolution_coeff_fit: Fit coefficients passed to
            `corrections.prepare_resolution_correction`.
        lines_coeff_fit: Fit coefficients passed to
            `corrections.prepare_lines_correction`.
        hcd_coeff_fit: Fit coefficients passed to
            `corrections.prepare_hcd_correction` for the DLA masking term.
        bal_coeff_fit: Fit coefficients passed to
            `corrections.prepare_hcd_correction` for the BAL AI masking
            term.
        continuum_coeff_fit: Fit coefficients passed to
            `corrections.prepare_cont_correction`.
        dla_completeness_coef (sequence): Per-redshift-bin coefficients
            passed to `hcd.rogers` for the DLA completeness correction; if
            a redshift bin index has no entry, the correction defaults to 1
            (no systematic).
        bal_completeness_coef (sequence): Per-redshift-bin coefficients
            passed to `hcd.rogers` for the BAL completeness correction; if
            a redshift bin index has no entry, the correction defaults to 1
            (no systematic).
        pk_low_z: `Pk1D`-like object to use instead of `pk` for redshift
            bins below `z_change`. Defaults to None (always use `pk`).
        z_change (float, optional): Redshift threshold below which
            `pk_low_z` is used instead of `pk`.
        **plot_args: Optional plotting overrides, e.g. `fontsize_x`,
            `fontsize_y`, `fontlegend`, `title_shift`, `title_yshift`,
            `size`, `title_size`, `figsize`, `ylim_top`, `ylim_bottom`,
            `n_subplots`, `color_map` (`"default"` or `"rainbow"`).

    Returns:
        None
    """
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 18)
    fontlegend = utils.return_key(plot_args, "fontlegend", 14)
    title_shift = utils.return_key(plot_args, "title_shift", 1.1)
    title_yshift = utils.return_key(plot_args, "title_yshift", 1.05)
    size = utils.return_key(plot_args, "size", 12)
    title_size = utils.return_key(plot_args, "title_size", 14)
    figsize = utils.return_key(plot_args, "figsize", (11, 16))
    ylim_top = utils.return_key(plot_args, "ylim_top", None)
    ylim_bottom = utils.return_key(plot_args, "ylim_bottom", None)
    n_subplots = utils.return_key(plot_args, "n_subplots", 10)

    zbins = pk.zbin[pk.zbin < zmax]

    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        colors = [f"C{i}" for i, z in enumerate(zbins)]
    elif color_map == "rainbow":
        colors = cm.rainbow(np.linspace(0, 1, len(zbins)))

    fig, ax = plt.subplots(n_subplots, 2, figsize=figsize, sharex=True)

    A_lines = corrections.prepare_lines_correction(zbins, lines_coeff_fit)
    A_hcd = corrections.prepare_hcd_correction(zbins, hcd_coeff_fit)
    A_bal = corrections.prepare_hcd_correction(zbins, bal_coeff_fit)
    A_cont = corrections.prepare_cont_correction(zbins, continuum_coeff_fit)
    A_reso = corrections.prepare_resolution_correction(zbins, resolution_coeff_fit)

    (
        wavenumber_systematics,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_bal,
        syste_continuum,
        syste_dla_completeness,
        syste_bal_completeness,
        syste_tot,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})

    for iz, z in enumerate(zbins):
        pk_to_plot = pk
        if pk_low_z is not None:
            if z < z_change:
                pk_to_plot = pk_low_z
        syste_tot[z] = []

        syste_reso[z] = (
            2 * pk_to_plot.k[z] ** 2 * delta_l[iz] * delta_delta_l[iz] * pk_to_plot.p[z]
        )
        syste_tot[z].append(syste_reso[z] ** 2)
        ax[0][1].plot(
            pk_to_plot.k[z],
            syste_reso[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[0][0].plot(
            pk_to_plot.k[z],
            syste_reso[z],
            color=colors[iz],
        )
        ax[0][0].set_title(
            "Resolution",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_resocorrection[z] = (
            0.3 * np.abs(A_reso[z](pk_to_plot.k[z]) - 1) * pk_to_plot.p[z]
        )
        syste_tot[z].append(syste_resocorrection[z] ** 2)
        ax[1][1].plot(
            pk_to_plot.k[z],
            syste_resocorrection[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[1][0].plot(
            pk_to_plot.k[z],
            syste_resocorrection[z],
            color=colors[iz],
        )
        ax[1][0].set_title(
            "Resolution Correction",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_sb[z] = pk_sb.err[z]
        syste_tot[z].append(syste_sb[z] ** 2)
        ax[2][1].plot(
            pk_to_plot.k[z],
            syste_sb[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[2][0].plot(
            pk_to_plot.k[z],
            syste_sb[z],
            color=colors[iz],
        )
        ax[2][0].set_title(
            "Metal power spectrum",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_continuum[z] = (
            0.3 * np.abs(A_cont[z](pk_to_plot.k[z]) - 1) * pk_to_plot.p[z]
        )
        syste_tot[z].append(syste_continuum[z] ** 2)
        ax[3][1].plot(
            pk_to_plot.k[z],
            syste_continuum[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[3][0].plot(
            pk_to_plot.k[z],
            syste_continuum[z],
            color=colors[iz],
        )
        ax[3][0].set_title(
            "Continuum fitting",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_lines[z] = 0.3 * np.abs(A_lines[z](pk_to_plot.k[z]) - 1) * pk_to_plot.p[z]
        syste_tot[z].append(syste_lines[z] ** 2)
        ax[4][1].plot(
            pk_to_plot.k[z],
            syste_lines[z] / pk_to_plot.err[z],
            label=r"$z = ${:1.1f}".format(z),
            color=colors[iz],
        )
        ax[4][0].plot(
            pk_to_plot.k[z],
            syste_lines[z],
            color=colors[iz],
        )
        ax[4][0].set_title(
            "Line masking",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_hcd[z] = 0.3 * np.abs(A_hcd[z](pk_to_plot.k[z]) - 1) * pk_to_plot.p[z]
        syste_tot[z].append(syste_hcd[z] ** 2)
        ax[5][1].plot(
            pk_to_plot.k[z],
            syste_hcd[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[5][0].plot(
            pk_to_plot.k[z],
            syste_hcd[z],
            color=colors[iz],
        )
        ax[5][0].set_title(
            "DLA masking",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )
        # 0.06 from mocks because 0.3 was overestimating the systematics
        syste_bal[z] = 0.06 * np.abs(A_bal[z](pk_to_plot.k[z]) - 1) * pk_to_plot.p[z]
        syste_tot[z].append(syste_bal[z] ** 2)
        ax[6][1].plot(
            pk_to_plot.k[z],
            syste_bal[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[6][0].plot(
            pk_to_plot.k[z],
            syste_bal[z],
            color=colors[iz],
        )
        ax[6][0].set_title(
            "BAL AI masking",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        if iz >= len(dla_completeness_coef):
            print(f"Redshift bin {z} have no dla completeness info")
            A_dla_completeness = np.poly1d(1)
        else:
            A_dla_completeness = hcd.rogers(
                z, pk_to_plot.k[z], *dla_completeness_coef[iz]
            )
        syste_dla_completeness[z] = (
            0.2 * np.abs(A_dla_completeness - 1) * pk_to_plot.p[z]
        )
        syste_tot[z].append(syste_dla_completeness[z] ** 2)
        ax[7][1].plot(
            pk_to_plot.k[z],
            syste_dla_completeness[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[7][0].plot(
            pk_to_plot.k[z],
            syste_dla_completeness[z],
            color=colors[iz],
        )
        ax[7][0].set_title(
            "DLA completeness",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        if iz >= len(bal_completeness_coef):
            print(f"Redshift bin {z} have no dla completeness info")
            A_bal_completeness = np.poly1d(1)
        else:
            A_bal_completeness = hcd.rogers(
                z, pk_to_plot.k[z], *bal_completeness_coef[iz]
            )
        syste_bal_completeness[z] = (
            0.15 * np.abs(A_bal_completeness - 1) * pk_to_plot.p[z]
        )
        syste_tot[z].append(syste_bal_completeness[z] ** 2)
        ax[8][1].plot(
            pk_to_plot.k[z],
            syste_bal_completeness[z] / pk_to_plot.err[z],
            color=colors[iz],
        )
        ax[8][0].plot(
            pk_to_plot.k[z],
            syste_bal_completeness[z],
            color=colors[iz],
        )
        ax[8][0].set_title(
            "BAL completeness",
            x=title_shift,
            y=title_yshift,
            fontsize=title_size,
        )

        syste_tot[z] = np.sqrt(np.sum(syste_tot[z], axis=0))
        ax[9][1].plot(
            pk_to_plot.k[z], syste_tot[z] / pk_to_plot.err[z], color=colors[iz]
        )
        ax[9][0].plot(pk_to_plot.k[z], syste_tot[z], color=colors[iz])
        ax[9][0].set_title("Total", x=title_shift, y=title_yshift, fontsize=title_size)
        wavenumber_systematics[z] = pk_to_plot.k[z]

    ax[0][0].set_xlim(kmin, kmax)

    ax[4][1].legend(
        loc=2, bbox_to_anchor=(1.04, 1.25), borderaxespad=0.0, fontsize=fontlegend
    )

    for i in range(n_subplots):
        ax[i][1].set_ylabel(
            r"$\sigma_{\mathrm{syst}} / \sigma_{\mathrm{stat}}$", fontsize=fontsize_y
        )
        ax[i][0].set_ylabel(r"$\sigma_{\mathrm{syst}}$", fontsize=fontsize_y)
        ax[i][0].set_yscale("log")
        ax[i][0].yaxis.set_tick_params(labelsize=size)
        ax[i][1].yaxis.set_tick_params(labelsize=size)

    if ylim_top is not None:
        for i in range(len(ylim_top)):
            ax[i][1].set_ylim(top=ylim_top[i][1])
            ax[i][0].set_ylim(top=ylim_top[i][0])
    if ylim_bottom is not None:
        for i in range(len(ylim_bottom)):
            ax[i][1].set_ylim(bottom=ylim_bottom[i][1])
            ax[i][0].set_ylim(bottom=ylim_bottom[i][0])
    if pk.velunits:
        ax[-1][0].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
        ax[-1][0].xaxis.set_tick_params(labelsize=size)
        ax[-1][1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
        ax[-1][1].xaxis.set_tick_params(labelsize=size)
    else:
        ax[-1][0].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
        ax[-1][0].xaxis.set_tick_params(labelsize=size)
        ax[-1][1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
        ax[-1][1].xaxis.set_tick_params(labelsize=size)

    fig.subplots_adjust(wspace=0.2, hspace=0.45, right=0.85)
    if pk.velunits:
        plt.savefig(f"{outname}_kms.pdf")
        name_file = f"{outname}_kms.pickle"
    else:
        plt.savefig(f"{outname}.pdf")
        name_file = f"{outname}.pickle"
    create_uncertainty_systematics_y1(
        wavenumber_systematics,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_bal,
        syste_continuum,
        syste_dla_completeness,
        syste_bal_completeness,
        syste_tot,
        name_file,
        pk.velunits,
        zbins,
    )


def plot_covariance(
    pk,
    zmax,
    kmax_AA,
    kmin_AA,
    out_name=None,
    out_points=None,
    use_boot=True,
    add_systematics=True,
    systematics_file=None,
    plot_correlation=True,
    pk_low_z=None,
    z_change=None,
    **plot_args,
):
    """Plot per-redshift-bin P1D covariance/correlation matrices as a grid of heatmaps.

    For each redshift bin below `zmax`, restricts the (bootstrap or
    analytic) covariance matrix to the `[kmin_AA, kmax_AA]` window
    (converted to s/km if `pk` is in velocity units), optionally adds
    systematic-error contributions (outer products of each systematic term
    loaded from `systematics_file`) to the diagonal-inflated matrix, then
    plots either the correlation matrix (covariance normalized by
    `sqrt(diag(cov))` outer product) or the raw covariance matrix as an
    imshow heatmap in a subplot grid. Optionally saves the figure and/or a
    flat text table of the plotted (k1, k2, covariance, correlation)
    points.

    Args:
        pk: `Pk1D`-like object exposing `zbin`, `velunits`, `k`, `boot_cov`,
            `cov` (all dict-like, keyed by redshift bin).
        zmax (float): Upper redshift bound (exclusive); only bins with
            `pk.zbin < zmax` are plotted.
        kmax_AA (float): Upper wavenumber bound in Angstrom^-1 (converted to
            s/km per bin if velocity units are used).
        kmin_AA (float): Lower wavenumber bound in Angstrom^-1 (converted to
            s/km per bin if velocity units are used).
        out_name (str, optional): If given, path (without extension) to
            save the figure as `.pdf`.
        out_points (str, optional): If given, base path (without extension)
            to save the plotted (z, k1, k2, covariance, correlation) points
            as a `.txt` table.
        use_boot (bool): If True, use `pk.boot_cov` (bootstrap covariance);
            otherwise use `pk.cov`. Defaults to True.
        add_systematics (bool): If True, add the systematic error terms
            (loaded from `systematics_file`) to the covariance matrix as
            outer products before computing the correlation. Defaults to
            True.
        systematics_file (str, optional): Path to a pickle file readable by
            `prepare_uncertainty_systematics`, required if
            `add_systematics` is True.
        plot_correlation (bool): If True, plot the normalized correlation
            matrix; otherwise plot the raw covariance matrix. Defaults to
            True.
        pk_low_z: `Pk1D`-like object to use instead of `pk` for redshift
            bins below `z_change`. Defaults to None (always use `pk`).
        z_change (float, optional): Redshift threshold below which
            `pk_low_z` is used instead of `pk`.
        **plot_args: Optional plotting overrides, e.g. `figsize`,
            `subplot_x`, `subplot_y`, `vmin`, `vmax`, `map_color`,
            `labelsize`, `titlesize`, `fontsize`, `cbar_prop`.

    Raises:
        ValueError: If `add_systematics` is True and `systematics_file` is
            None.

    Returns:
        None
    """
    figsize = utils.return_key(plot_args, "figsize", (20, 20))
    subplot_x = utils.return_key(plot_args, "subplot_x", 4)
    subplot_y = utils.return_key(plot_args, "subplot_y", 4)
    vmin = utils.return_key(plot_args, "vmin", -1)
    vmax = utils.return_key(plot_args, "vmax", 1)
    cmap = utils.return_key(plot_args, "map_color", "seismic")
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    titlesize = utils.return_key(plot_args, "titlesize", 14)
    fontsize = utils.return_key(plot_args, "fontsize", 14)
    cbar_prop = utils.return_key(plot_args, "cbar_prop", [0.92, 0.1, 0.02, 0.4])

    zbins = pk.zbin[pk.zbin < zmax]

    k1_arr, k2_arr, cov_mat_arr, corr_mat_arr, z_arr = [], [], [], [], []
    if add_systematics & (systematics_file is not None):
        (
            _,
            list_systematics,
            _,
        ) = prepare_uncertainty_systematics(systematics_file)

    elif add_systematics & (systematics_file is None):
        raise ValueError("You need to provide a systematics file to add systematics")

    fig = plt.figure(figsize=figsize)
    for j, z in enumerate(zbins):
        pk_to_plot = pk
        if pk_low_z is not None:
            if z < z_change:
                pk_to_plot = pk_low_z
        if pk_to_plot.velunits:
            kmax = float(utils.kAAtokskm(kmax_AA, z=z))
            kmin = float(utils.kAAtokskm(kmin_AA, z=z))
        else:
            kmax = kmax_AA
            kmin = kmin_AA

        mask = (pk_to_plot.k[z] > kmin) & (pk_to_plot.k[z] < kmax)

        kmin_plot = np.min(pk_to_plot.k[z][mask])
        kmax_plot = np.max(pk_to_plot.k[z][mask])
        extent = [kmin_plot, kmax_plot, kmax_plot, kmin_plot]
        if use_boot:
            cov_matrix = pk_to_plot.boot_cov[z].reshape(
                len(pk_to_plot.k[z]), len(pk_to_plot.k[z])
            )
        else:
            cov_matrix = pk_to_plot.cov[z].reshape(
                len(pk_to_plot.k[z]), len(pk_to_plot.k[z])
            )
        cov_matrix_cut = cov_matrix[
            np.ix_(np.argwhere(mask)[:, 0], np.argwhere(mask)[:, 0])
        ]

        if add_systematics:
            for i in range(len(list_systematics)):
                cov_sys = np.outer(
                    list_systematics[i][z][mask], list_systematics[i][z][mask]
                )
                cov_matrix_cut = cov_matrix_cut + cov_sys

        mean_k1 = np.array(
            [pk_to_plot.k[z][mask] for i in range(len(pk_to_plot.k[z][mask]))]
        ).T
        mean_k2 = np.array(
            [pk_to_plot.k[z][mask] for i in range(len(pk_to_plot.k[z][mask]))]
        )

        v = np.sqrt(np.diag(cov_matrix_cut))
        outer_v = np.outer(v, v)
        if plot_correlation:
            corr_mat = cov_matrix_cut / outer_v
        else:
            corr_mat = cov_matrix_cut

        ax = fig.add_subplot(subplot_y, subplot_x, j + 1)
        ax.set_title(f"z = {z}", fontsize=titlesize)
        im = ax.imshow(corr_mat, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        if pk.velunits:
            ax.set_xlabel(
                r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$",
                fontsize=fontsize,
            )
            ax.set_ylabel(
                r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$",
                fontsize=fontsize,
            )
            ax.xaxis.set_tick_params(labelsize=labelsize)
            ax.yaxis.set_tick_params(labelsize=labelsize)
        else:
            ax.set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize)
            ax.set_ylabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize)
            ax.xaxis.set_tick_params(labelsize=labelsize)
            ax.yaxis.set_tick_params(labelsize=labelsize)

        z_arr.append(np.full(np.ravel(mean_k1).shape, z))
        k1_arr.append(np.ravel(mean_k1))
        k2_arr.append(np.ravel(mean_k2))
        cov_mat_arr.append(np.ravel(cov_matrix_cut))
        corr_mat_arr.append(np.ravel(corr_mat))
    cax = fig.add_axes(cbar_prop)
    fig.colorbar(im, cax=cax)
    cax.yaxis.set_tick_params(labelsize=labelsize)
    cax.set_ylabel("Correlation matrix", fontsize=fontsize)

    z_arr = np.concatenate(z_arr, axis=0)
    k1_arr = np.concatenate(k1_arr, axis=0)
    k2_arr = np.concatenate(k2_arr, axis=0)
    cov_mat_arr = np.concatenate(cov_mat_arr, axis=0)
    corr_mat_arr = np.concatenate(corr_mat_arr, axis=0)

    fig.tight_layout()
    if out_name is not None:
        fig.savefig(f"{out_name}.pdf", format="pdf")
    if out_points is not None:
        if pk.velunits:
            header = "REDSHIFT & WAVENUMBER 1 [s.km^-1] & WAVENUMBER 2 [s.km^-1] & COVARIANCE MATRIX & CORRELATION MATRIX"
        else:
            header = "REDSHIFT & WAVENUMBER 1 [Ang^-1] & WAVENUMBER 2 [Ang^-1] & COVARIANCE MATRIX & CORRELATION MATRIX"
        np.savetxt(
            f"{out_points}.txt",
            np.transpose(np.stack([z_arr, k1_arr, k2_arr, cov_mat_arr, corr_mat_arr])),
            header=header,
        )
