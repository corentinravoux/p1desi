"""Noise estimation and correction diagnostics for the DESI Lyman-alpha 1D power spectrum (P1D).

This module provides plotting and computation utilities to compare the raw
P1D (``p_raw``), the pipeline noise estimate (``p_noise``) and the
exposure-difference noise estimate (``p_diff``) of a :class:`p1desi.pk_io.Pk1D`-like
object, per redshift bin (``pk.zbin``) and as a function of wavenumber
``k`` (in Angstrom^-1, unless ``pk.velunits`` is set, in which case units are
s/km). It also exposes helpers to compute the noise asymptote (systematic
offset ``alpha`` and multiplicative ratio ``beta``) at high-k, using the
mean-over-redshift (:class:`p1desi.pk_io.MeanPkZ`) and mean-over-wavenumber
(:class:`p1desi.pk_io.MeanPkK`) helper classes from :mod:`p1desi.pk_io`.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

from p1desi import pk_io, utils


def plot_noise_power_ratio(
    pk,
    use_diff=False,
    out_name=None,
    out_points=None,
    plot_asymptote=True,
    k_asymptote=3.10,
    zmax=None,
    **kwargs,
):
    """Plot raw vs. noise power, their difference, and their ratio per redshift bin.

    Builds a 3-panel figure (raw & noise P(k); raw-minus-noise; noise/raw
    ratio), overlaying one curve per redshift bin of ``pk.zbin``, and
    optionally annotates the high-k noise asymptote (see
    :func:`p1desi.pk_io.MeanPkZ.compute_noise_asymptopte`).

    Args:
        pk: A ``Pk1D``-like object (see :mod:`p1desi.pk_io`) exposing
            ``zbin``, ``k``, ``p_raw``, ``p_noise`` and ``p_diff`` dicts
            keyed by redshift.
        use_diff (bool): If True, use ``pk.p_diff`` (exposure-difference
            noise estimate) instead of ``pk.p_noise`` (pipeline noise
            estimate). Defaults to False.
        out_name (str, optional): If given, base path (without extension)
            used to save the figure as ``f"{out_name}.pdf"``.
        out_points (optional): If not None, save the underlying arrays
            (redshift, k, raw power, noise power) to ``f"{out_name}.txt"``.
        plot_asymptote (bool): If True, compute and display the noise
            asymptote coefficients alpha and beta on the difference and
            ratio panels. Defaults to True.
        k_asymptote (float): Wavenumber threshold (in the units of
            ``pk.k``) above which the asymptote is estimated. Defaults to
            3.10.
        zmax (float, optional): If given, only redshift bins with
            ``z <= zmax`` are plotted.
        **kwargs: Optional style/layout overrides, including ``figsize``,
            ``ncol_legend``, ``y_min``, ``y_max``, ``diff_y_min``,
            ``diff_y_max``, ``ratio_y_min``, ``ratio_y_max``,
            ``legend_size``, ``fontsize``, ``ticks_size``,
            ``plot_velunits`` and ``cmap``.

    Returns:
        None
    """
    figsize = utils.return_key(kwargs, "figsize", (8, 10))
    ncol_legend = utils.return_key(kwargs, "ncol_legend", 2)
    y_min = utils.return_key(kwargs, "y_min", 0.025)
    y_max = utils.return_key(kwargs, "y_max", 0.05)
    diff_y_min = utils.return_key(kwargs, "diff_y_min", -0.01)
    diff_y_max = utils.return_key(kwargs, "diff_y_max", 0.15)
    ratio_y_min = utils.return_key(kwargs, "ratio_y_min", 0.6)
    ratio_y_max = utils.return_key(kwargs, "ratio_y_max", 1.05)
    legend_size = utils.return_key(kwargs, "legend_size", 15)
    fontsize = utils.return_key(kwargs, "fontsize", 16)
    ticks_size = utils.return_key(kwargs, "ticks_size", 15)
    plot_velunits = utils.return_key(kwargs, "plot_velunits", True)
    cmap = utils.return_key(kwargs, "cmap", "rainbow")
    if cmap == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin)))
    else:
        color = [f"C{i}" for i in range(len(pk.zbin))]

    if use_diff:
        noise = pk.p_diff
        labelnoise = "diff"
    else:
        noise = pk.p_noise
        labelnoise = "pipeline"

    fig, ax = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[2, 1, 1])
    )
    z_arr, k_arr, praw_arr, pnoise_arr = [], [], [], []
    zbins = []
    for i, z in enumerate(pk.zbin):
        if zmax is not None:
            if z > zmax:
                continue
        zbins.append(z)
        z_arr.append([z for i in range(len(pk.k[z]))])
        k_arr.append(pk.k[z])
        praw_arr.append(pk.p_raw[z])
        pnoise_arr.append(noise[z])
        ax[0].plot(pk.k[z], noise[z], ls="--", color=color[i])
        ax[0].plot(pk.k[z], pk.p_raw[z], color=color[i])
        ax[1].plot(pk.k[z], pk.p_raw[z] - noise[z], label=f"{z:.1f}", color=color[i])
        ax[2].plot(pk.k[z], noise[z] / pk.p_raw[z], label=f"{z:.1f}", color=color[i])

    z_arr, k_arr, praw_arr, pnoise_arr = (
        np.concatenate(z_arr),
        np.concatenate(k_arr),
        np.concatenate(praw_arr),
        np.concatenate(pnoise_arr),
    )

    ax[0].set_ylabel(r"$P~[\AA]$", fontsize=fontsize)

    legend_elements = [
        Line2D(
            [],
            [],
            color=color[i],
            marker=None,
            linestyle="-",
            label=f"z = {zbins[i]:.1f}",
        )
        for i in range(len(zbins))
    ]
    legend_elements = legend_elements + [
        Line2D(
            [], [], color="k", marker=None, linestyle="-", label=r"$P_{\mathrm{raw}}$"
        ),
        Line2D(
            [],
            [],
            color="k",
            marker=None,
            linestyle="--",
            label="$P_{\mathrm{" + labelnoise + "}}$",
        ),
    ]
    ax[0].legend(
        handles=legend_elements,
        fontsize=legend_size,
        ncol=ncol_legend,
        loc="upper left",
    )

    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(diff_y_min, diff_y_max)
    ax[2].set_ylim(ratio_y_min, ratio_y_max)

    ax[1].set_ylabel(
        r"$P_{\mathrm{raw}} - P_{\mathrm{" + labelnoise + "}}[\AA]$", fontsize=fontsize
    )
    ax[2].set_ylabel(
        r"$P_{\mathrm{" + labelnoise + "}}/P_{\mathrm{raw}}$", fontsize=fontsize
    )
    ax[2].set_xlabel("$k~[\AA^{-1}]$", fontsize=fontsize)

    for i in [0, 1, 2]:
        ax[i].xaxis.set_tick_params(labelsize=ticks_size)
        ax[i].yaxis.set_tick_params(labelsize=ticks_size)
        ax[i].margins(x=0)

    if plot_asymptote:
        mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk, zmax=zmax)
        alpha, beta = mean_pk_z.compute_noise_asymptopte(k_asymptote, use_diff=use_diff)
        empty_patch = [
            mpatches.Patch(
                color="none",
                label=r"$\alpha =$" + f" {np.around(alpha,5)} " + "$\AA$         ",
            )
        ]
        ax[1].legend(handles=empty_patch, fontsize=legend_size)
        empty_patch = [
            mpatches.Patch(
                color="none",
                label=r"$\beta =$" + f" {np.around(beta,3)}" + "          ",
            )
        ]
        ax[2].legend(handles=empty_patch, fontsize=legend_size)

    if plot_velunits:
        utils.place_k_speed_unit_axis(fig, ax[0], fontsize=fontsize)

    fig.tight_layout()
    if out_name is not None:
        fig.savefig(f"{out_name}.pdf", format="pdf")
    if out_points is not None:
        np.savetxt(
            f"{out_name}.txt",
            np.transpose(np.stack([z_arr, k_arr, praw_arr, pnoise_arr])),
            header="REDSHIFT & WAVENUMBER [Ang^-1] & RAW POWER SPECTRA & NOISE POWER SPECTRA",
        )


def plot_noise_power_ratio_bis(
    pk,
    use_diff=False,
    out_name=None,
    out_points=None,
    plot_asymptote=True,
    k_asymptote=3.10,
    zmax=None,
    **kwargs,
):
    """Plot raw vs. noise power and their difference per redshift bin, in a 2-panel layout.

    Variant of :func:`plot_noise_power_ratio` restricted to two side-by-side
    panels (raw & noise P(k); raw-minus-noise), without the ratio panel.

    Args:
        pk: A ``Pk1D``-like object (see :mod:`p1desi.pk_io`) exposing
            ``zbin``, ``k``, ``p_raw``, ``p_noise`` and ``p_diff`` dicts
            keyed by redshift.
        use_diff (bool): If True, use ``pk.p_diff`` (exposure-difference
            noise estimate) instead of ``pk.p_noise`` (pipeline noise
            estimate). Defaults to False.
        out_name (str, optional): If given, base path (without extension)
            used to save the figure as ``f"{out_name}.pdf"``.
        out_points (optional): If not None, save the underlying arrays
            (redshift, k, raw power, noise power) to ``f"{out_name}.txt"``.
        plot_asymptote (bool): If True, compute and display the noise
            asymptote coefficient alpha on the difference panel (see
            :func:`p1desi.pk_io.MeanPkZ.compute_noise_asymptopte`).
            Defaults to True.
        k_asymptote (float): Wavenumber threshold (in the units of
            ``pk.k``) above which the asymptote is estimated. Defaults to
            3.10.
        zmax (float, optional): If given, only redshift bins with
            ``z <= zmax`` are plotted.
        **kwargs: Optional style/layout overrides, including ``figsize``,
            ``ncol_legend``, ``y_min``, ``y_max``, ``diff_y_min``,
            ``diff_y_max``, ``legend_size``, ``fontsize``, ``ticks_size``,
            ``cmap`` and ``force_label`` (override the noise-curve legend
            label).

    Returns:
        None
    """
    figsize = utils.return_key(kwargs, "figsize", (11, 5))
    ncol_legend = utils.return_key(kwargs, "ncol_legend", 2)
    y_min = utils.return_key(kwargs, "y_min", 0.025)
    y_max = utils.return_key(kwargs, "y_max", 0.05)
    diff_y_min = utils.return_key(kwargs, "diff_y_min", -0.01)
    diff_y_max = utils.return_key(kwargs, "diff_y_max", 0.15)
    legend_size = utils.return_key(kwargs, "legend_size", 15)
    fontsize = utils.return_key(kwargs, "fontsize", 16)
    ticks_size = utils.return_key(kwargs, "ticks_size", 15)
    cmap = utils.return_key(kwargs, "cmap", "rainbow")
    force_label = utils.return_key(kwargs, "force_label", None)
    if cmap == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin)))
    else:
        color = [f"C{i}" for i in range(len(pk.zbin))]

    if use_diff:
        noise = pk.p_diff
        labelnoise = "diff"
    else:
        noise = pk.p_noise
        labelnoise = "pipeline"
    if force_label is not None:
        labelnoise = force_label

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    z_arr, k_arr, praw_arr, pnoise_arr = [], [], [], []
    zbins = []
    for i, z in enumerate(pk.zbin):
        if zmax is not None:
            if z > zmax:
                continue
        zbins.append(z)
        z_arr.append([z for i in range(len(pk.k[z]))])
        k_arr.append(pk.k[z])
        praw_arr.append(pk.p_raw[z])
        pnoise_arr.append(noise[z])
        ax[0].plot(pk.k[z], noise[z], ls="--", color=color[i])
        ax[0].plot(pk.k[z], pk.p_raw[z], color=color[i])
        ax[1].plot(pk.k[z], pk.p_raw[z] - noise[z], label=f"{z:.1f}", color=color[i])

    z_arr, k_arr, praw_arr, pnoise_arr = (
        np.concatenate(z_arr),
        np.concatenate(k_arr),
        np.concatenate(praw_arr),
        np.concatenate(pnoise_arr),
    )

    ax[0].set_ylabel(r"$P~[\AA]$", fontsize=fontsize)

    legend_elements = [
        Line2D(
            [],
            [],
            color=color[i],
            marker=None,
            linestyle="-",
            label=f"z = {zbins[i]:.1f}",
        )
        for i in range(len(zbins))
    ]
    legend_elements = legend_elements + [
        Line2D(
            [], [], color="k", marker=None, linestyle="-", label=r"$P_{\mathrm{raw}}$"
        ),
        Line2D(
            [],
            [],
            color="k",
            marker=None,
            linestyle="--",
            label="$P_{\mathrm{" + labelnoise + "}}$",
        ),
    ]
    ax[0].legend(
        handles=legend_elements,
        fontsize=legend_size,
        ncol=ncol_legend,
        loc="upper left",
    )

    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(diff_y_min, diff_y_max)

    ax[1].set_ylabel(
        r"$P_{\mathrm{raw}} - P_{\mathrm{" + labelnoise + "}}[\AA]$", fontsize=fontsize
    )

    ax[0].set_xlabel("$k~[\AA^{-1}]$", fontsize=fontsize)
    ax[1].set_xlabel("$k~[\AA^{-1}]$", fontsize=fontsize)

    for i in [0, 1]:
        ax[i].xaxis.set_tick_params(labelsize=ticks_size)
        ax[i].yaxis.set_tick_params(labelsize=ticks_size)
        ax[i].margins(x=0)

    if plot_asymptote:
        mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk, zmax=zmax)
        alpha, _ = mean_pk_z.compute_noise_asymptopte(k_asymptote, use_diff=use_diff)
        empty_patch = [
            mpatches.Patch(
                color="none",
                label=r"$\alpha =$" + f" {np.around(alpha,5)} " + "$\AA$         ",
            )
        ]
        ax[1].legend(handles=empty_patch, fontsize=legend_size)

    fig.tight_layout()
    if out_name is not None:
        fig.savefig(f"{out_name}.pdf", format="pdf")
    if out_points is not None:
        np.savetxt(
            f"{out_name}.txt",
            np.transpose(np.stack([z_arr, k_arr, praw_arr, pnoise_arr])),
            header="REDSHIFT & WAVENUMBER [Ang^-1] & RAW POWER SPECTRA & NOISE POWER SPECTRA",
        )


def plot_noise_comparison(
    pk,
    out_name=None,
    out_points=None,
    **kwargs,
):
    """Plot pipeline vs. exposure-difference noise power, per redshift bin.

    Builds a 3-panel figure comparing ``pk.p_noise`` (pipeline noise
    estimate) and ``pk.p_diff`` (exposure-difference noise estimate)
    directly: their curves, their difference, and their ratio, one curve
    per redshift bin of ``pk.zbin``.

    Args:
        pk: A ``Pk1D``-like object (see :mod:`p1desi.pk_io`) exposing
            ``zbin``, ``k``, ``p_noise`` and ``p_diff`` dicts keyed by
            redshift.
        out_name (str, optional): If given, base path (without extension)
            used to save the figure as ``f"{out_name}.pdf"``.
        out_points (optional): If not None, save the underlying arrays
            (redshift, k, pipeline noise power, diff noise power) to
            ``f"{out_name}.txt"``.
        **kwargs: Optional style/layout overrides, including ``figsize``,
            ``ncol_legend``, ``y_min``, ``y_max``, ``diff_y_min``,
            ``diff_y_max``, ``ratio_y_min``, ``ratio_y_max``,
            ``legend_size``, ``fontsize``, ``ticks_size``,
            ``plot_velunits`` and ``cmap``.

    Returns:
        None
    """
    figsize = utils.return_key(kwargs, "figsize", (8, 10))
    ncol_legend = utils.return_key(kwargs, "ncol_legend", 2)
    y_min = utils.return_key(kwargs, "y_min", 0.025)
    y_max = utils.return_key(kwargs, "y_max", 0.05)
    diff_y_min = utils.return_key(kwargs, "diff_y_min", -0.1)
    diff_y_max = utils.return_key(kwargs, "diff_y_max", 0.1)
    ratio_y_min = utils.return_key(kwargs, "ratio_y_min", 0.95)
    ratio_y_max = utils.return_key(kwargs, "ratio_y_max", 1.05)
    legend_size = utils.return_key(kwargs, "legend_size", 15)
    fontsize = utils.return_key(kwargs, "fontsize", 16)
    ticks_size = utils.return_key(kwargs, "ticks_size", 15)
    plot_velunits = utils.return_key(kwargs, "plot_velunits", True)
    cmap = utils.return_key(kwargs, "cmap", "rainbow")
    if cmap == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin)))
    else:
        color = [f"C{i}" for i in range(len(pk.zbin))]

    fig, ax = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[2, 1, 1])
    )
    z_arr, k_arr, pnoise_arr, pdiff_arr = [], [], [], []

    for i, z in enumerate(pk.zbin):
        z_arr.append([z for i in range(len(pk.k[z]))])
        k_arr.append(pk.k[z])
        pnoise_arr.append(pk.p_noise[z])
        pdiff_arr.append(pk.p_diff[z])
        ax[0].plot(pk.k[z], pk.p_noise[z], color=color[i])
        ax[0].plot(pk.k[z], pk.p_diff[z], ls="--", color=color[i])
        ax[1].plot(
            pk.k[z], pk.p_noise[z] - pk.p_diff[z], label=f"{z:.1f}", color=color[i]
        )
        ax[2].plot(
            pk.k[z], pk.p_noise[z] / pk.p_diff[z], label=f"{z:.1f}", color=color[i]
        )

    z_arr, k_arr, pnoise_arr, pdiff_arr = (
        np.concatenate(z_arr),
        np.concatenate(k_arr),
        np.concatenate(pnoise_arr),
        np.concatenate(pdiff_arr),
    )

    ax[0].set_ylabel(r"$P~[\AA]$", fontsize=fontsize)

    legend_elements = [
        Line2D(
            [],
            [],
            color=color[i],
            marker=None,
            linestyle="-",
            label=f"z = {pk.zbin[i]:.1f}",
        )
        for i in range(len(pk.zbin))
    ]

    legend_elements = legend_elements + [
        Line2D(
            [],
            [],
            color="k",
            marker=None,
            linestyle="-",
            label=r"$P_{\mathrm{pipeline}}$",
        ),
        Line2D(
            [],
            [],
            color="k",
            marker=None,
            linestyle="--",
            label="$P_{\mathrm{diff}}$",
        ),
    ]

    ax[0].legend(
        handles=legend_elements,
        fontsize=legend_size,
        ncol=ncol_legend,
        loc="upper left",
    )

    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(diff_y_min, diff_y_max)
    ax[2].set_ylim(ratio_y_min, ratio_y_max)

    ax[1].set_ylabel(
        r"$P_{\mathrm{pipeline}} - P_{\mathrm{diff}}[\AA]$", fontsize=fontsize
    )
    ax[2].set_ylabel(r"$P_{\mathrm{pipeline}}/P_{\mathrm{diff}}$", fontsize=fontsize)
    ax[2].set_xlabel("$k~[\AA^{-1}]$", fontsize=fontsize)

    for i in [0, 1, 2]:
        ax[i].xaxis.set_tick_params(labelsize=ticks_size)
        ax[i].yaxis.set_tick_params(labelsize=ticks_size)
        ax[i].margins(x=0)

    if plot_velunits:
        utils.place_k_speed_unit_axis(fig, ax[0], fontsize=fontsize)

    fig.tight_layout()
    if out_name is not None:
        fig.savefig(f"{out_name}.pdf", format="pdf")
    if out_points is not None:
        np.savetxt(
            f"{out_name}.txt",
            np.transpose(np.stack([z_arr, k_arr, pnoise_arr, pdiff_arr])),
            header="REDSHIFT & WAVENUMBER [Ang^-1] & PIPELINE POWER SPECTRA & DIFF POWER SPECTRA",
        )


# Mean redshift functions


def plot_noise_power_ratio_meanz(
    pk,
    use_diff=False,
    out_name=None,
    out_points=None,
    plot_asymptote=True,
    k_asymptote=3.10,
    zmax=None,
    **kwargs,
):
    """Plot the redshift-averaged raw vs. noise power, difference, and ratio.

    Same purpose as :func:`plot_noise_power_ratio` but operating on the
    redshift-mean power spectrum computed via
    :meth:`p1desi.pk_io.MeanPkZ.init_from_pk`, producing a single averaged
    curve per panel instead of one curve per redshift bin.

    Args:
        pk: A ``Pk1D``-like object (see :mod:`p1desi.pk_io`) passed to
            :meth:`p1desi.pk_io.MeanPkZ.init_from_pk` to compute the
            redshift-averaged raw and noise power spectra.
        use_diff (bool): If True, use the averaged ``p_diff``
            (exposure-difference noise estimate) instead of the averaged
            ``p_noise`` (pipeline noise estimate). Defaults to False.
        out_name (str, optional): If given, base path (without extension)
            used to save the figure as ``f"{out_name}.pdf"``.
        out_points (optional): If not None, save the underlying arrays
            (k, raw power, noise power) to ``f"{out_name}.txt"``.
        plot_asymptote (bool): If True, compute and display the noise
            asymptote coefficients alpha and beta on the difference and
            ratio panels (see
            :meth:`p1desi.pk_io.MeanPkZ.compute_noise_asymptopte`).
            Defaults to True.
        k_asymptote (float): Wavenumber threshold (in the units of
            ``mean_pk_z.k``) above which the asymptote is estimated.
            Defaults to 3.10.
        zmax (float, optional): If given, only redshift bins with
            ``z <= zmax`` are averaged over.
        **kwargs: Optional style/layout overrides, including ``figsize``,
            ``y_min``, ``y_max``, ``diff_y_min``, ``diff_y_max``,
            ``ratio_y_min``, ``ratio_y_max``, ``legend_size``,
            ``fontsize``, ``ticks_size`` and ``plot_velunits``.

    Returns:
        None
    """
    figsize = utils.return_key(kwargs, "figsize", (8, 10))
    y_min = utils.return_key(kwargs, "y_min", 0.025)
    y_max = utils.return_key(kwargs, "y_max", 0.05)
    diff_y_min = utils.return_key(kwargs, "diff_y_min", -0.01)
    diff_y_max = utils.return_key(kwargs, "diff_y_max", 0.15)
    ratio_y_min = utils.return_key(kwargs, "ratio_y_min", 0.6)
    ratio_y_max = utils.return_key(kwargs, "ratio_y_max", 1.05)
    legend_size = utils.return_key(kwargs, "legend_size", 15)
    fontsize = utils.return_key(kwargs, "fontsize", 16)
    ticks_size = utils.return_key(kwargs, "ticks_size", 15)
    plot_velunits = utils.return_key(kwargs, "plot_velunits", True)

    mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk, zmax=zmax)

    if use_diff:
        noise = mean_pk_z.p_diff
        labelnoise = "diff"
    else:
        noise = mean_pk_z.p_noise
        labelnoise = "pipeline"

    fig, ax = plt.subplots(
        3,
        1,
        figsize=figsize,
    )

    ax[0].plot(mean_pk_z.k, noise, ls="--")
    ax[0].plot(mean_pk_z.k, mean_pk_z.p_raw)
    ax[1].plot(mean_pk_z.k, mean_pk_z.p_raw - noise)
    ax[2].plot(mean_pk_z.k, noise / mean_pk_z.p_raw)

    ax[0].set_ylabel(r"$P~[\AA]$", fontsize=fontsize)

    ax[0].legend(
        [labelnoise, "raw"],
        fontsize=legend_size,
    )

    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(diff_y_min, diff_y_max)
    ax[2].set_ylim(ratio_y_min, ratio_y_max)
    ax[1].set_ylabel(
        r"$P_{\mathrm{raw}} - P_{\mathrm{" + labelnoise + "}}[\AA]$", fontsize=fontsize
    )
    ax[2].set_ylabel(
        r"$P_{\mathrm{" + labelnoise + "}}/P_{\mathrm{raw}}$", fontsize=fontsize
    )
    ax[2].set_xlabel("$k~[\AA^{-1}]$", fontsize=fontsize)

    for i in [0, 1, 2]:
        ax[i].xaxis.set_tick_params(labelsize=ticks_size)
        ax[i].yaxis.set_tick_params(labelsize=ticks_size)
        ax[i].margins(x=0)

    if plot_asymptote:
        mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk, zmax=zmax)
        alpha, beta = mean_pk_z.compute_noise_asymptopte(k_asymptote, use_diff=use_diff)
        empty_patch = [
            mpatches.Patch(
                color="none",
                label=r"$\alpha =$" + f" {np.around(alpha,5)} " + "$\AA$         ",
            )
        ]
        ax[1].legend(handles=empty_patch, fontsize=legend_size)
        empty_patch = [
            mpatches.Patch(
                color="none",
                label=r"$\beta =$" + f" {np.around(beta,3)}" + "          ",
            )
        ]
        ax[2].legend(handles=empty_patch, fontsize=legend_size)

    if plot_velunits:
        utils.place_k_speed_unit_axis(fig, ax[0], fontsize=fontsize)

    fig.tight_layout()
    if out_name is not None:
        fig.savefig(f"{out_name}.pdf", format="pdf")
    if out_points is not None:
        np.savetxt(
            f"{out_name}.txt",
            np.transpose(np.stack([mean_pk_z.k, mean_pk_z.p_raw, noise])),
            header="REDSHIFT & WAVENUMBER [Ang^-1] & RAW POWER SPECTRA & NOISE POWER SPECTRA",
        )


def compute_asymptote(
    pk,
    k_asymptote,
    zmax=None,
):
    """Compute the high-k noise asymptote coefficients for both noise estimators.

    Builds the redshift-averaged power spectrum via
    :meth:`p1desi.pk_io.MeanPkZ.init_from_pk`, then estimates the additive
    offset (alpha) and multiplicative ratio (beta) between the raw power
    and each of the pipeline-noise and diff-noise estimates above
    ``k_asymptote``, using
    :meth:`p1desi.pk_io.MeanPkZ.compute_noise_asymptopte`.

    Args:
        pk: A ``Pk1D``-like object (see :mod:`p1desi.pk_io`) exposing
            ``zbin``, ``k``, ``p_raw``, ``p_noise`` and ``p_diff``.
        k_asymptote (float): Wavenumber threshold (in the units of
            ``pk.k``, i.e. Angstrom^-1 or s/km depending on
            ``pk.velunits``) above which the asymptote is estimated.
        zmax (float, optional): If given, only redshift bins with
            ``z <= zmax`` are averaged over.

    Returns:
        tuple: ``(alpha_pipeline, beta_pipeline, alpha_diff, beta_diff)``,
        the additive offset and multiplicative ratio computed with the
        pipeline noise estimate (``use_diff=False``) and with the
        exposure-difference noise estimate (``use_diff=True``)
        respectively.
    """
    mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk, zmax=zmax)
    alpha_pipeline, beta_pipeline = mean_pk_z.compute_noise_asymptopte(
        k_asymptote, use_diff=False
    )
    alpha_diff, beta_diff = mean_pk_z.compute_noise_asymptopte(
        k_asymptote, use_diff=True
    )
    return alpha_pipeline, beta_pipeline, alpha_diff, beta_diff


# Mean wavenumber functions


def plot_noise_power_meank(
    pk,
    kmax=None,
    out_name=None,
    out_points=None,
    **kwargs,
):
    """Plot the wavenumber-averaged noise power and noise ratio vs. redshift.

    Builds a 2-panel figure showing, as a function of redshift ``z``: the
    k-averaged pipeline noise (``p_noise``) and diff noise (``p_diff``)
    power with their standard errors, and their ratio
    (``p_noise / p_diff``), using
    :meth:`p1desi.pk_io.MeanPkK.init_from_pk` and
    :meth:`p1desi.pk_io.MeanPkK.compute_additional_stats`.

    Args:
        pk: A ``Pk1D``-like object (see :mod:`p1desi.pk_io`) exposing
            ``zbin``, ``k``, ``p_noise`` and ``p_diff`` dicts keyed by
            redshift.
        kmax (float, optional): If given, only wavenumbers ``k < kmax``
            (units of ``pk.k``) are included in the per-redshift average.
        out_name (str, optional): If given, base path (without extension)
            used to save the figure as ``f"{out_name}.pdf"``.
        out_points (optional): If not None, save the underlying arrays
            (redshift, noise power & its error, diff power & its error,
            ratio & its error) to ``f"{out_name}.txt"``.
        **kwargs: Optional style/layout overrides, including ``figsize``,
            ``ax`` (reuse an existing pair of axes instead of creating a
            new figure), ``color``, ``marker``, ``y_min``, ``y_max``,
            ``ratio_y_min``, ``ratio_y_max``, ``legend_size``,
            ``fontsize`` and ``ticks_size``.

    Returns:
        None
    """
    figsize = utils.return_key(kwargs, "figsize", (8, 6))
    ax = utils.return_key(kwargs, "ax", None)
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    color = utils.return_key(kwargs, "color", "C0")
    marker = utils.return_key(kwargs, "marker", "*")

    y_min = utils.return_key(kwargs, "y_min", 0.025)
    y_max = utils.return_key(kwargs, "y_max", 0.05)
    ratio_y_min = utils.return_key(kwargs, "ratio_y_min", 0.6)
    ratio_y_max = utils.return_key(kwargs, "ratio_y_max", 1.05)
    legend_size = utils.return_key(kwargs, "legend_size", 15)
    fontsize = utils.return_key(kwargs, "fontsize", 16)
    ticks_size = utils.return_key(kwargs, "ticks_size", 15)

    mean_pk_k = pk_io.MeanPkK.init_from_pk(pk, kmax=kmax)
    mean_pk_k.compute_additional_stats(pk, kmax=kmax)

    ax[0].errorbar(
        mean_pk_k.zbin,
        [mean_pk_k.p_noise[z] for z in mean_pk_k.zbin],
        [mean_pk_k.err_noise[z] for z in mean_pk_k.zbin],
        marker=marker,
        color=color,
        linestyle="None",
    )
    ax[0].errorbar(
        mean_pk_k.zbin,
        [mean_pk_k.p_diff[z] for z in mean_pk_k.zbin],
        [mean_pk_k.err_diff[z] for z in mean_pk_k.zbin],
        marker=marker,
        markerfacecolor="None",
        markeredgecolor=color,
        linestyle="None",
    )

    ax[1].errorbar(
        mean_pk_k.zbin,
        [mean_pk_k.noiseoverdiff[z] for z in mean_pk_k.zbin],
        [mean_pk_k.err_noiseoverdiff[z] for z in mean_pk_k.zbin],
        marker=marker,
        color=color,
    )

    legend_elements = [
        Line2D(
            [],
            [],
            marker=marker,
            color=color,
            linestyle="None",
            label="$P_{\mathrm{pipeline}}$",
        ),
        Line2D(
            [],
            [],
            marker=marker,
            markerfacecolor="None",
            markeredgecolor=color,
            linestyle="None",
            label="$P_{\mathrm{diff}}$",
        ),
    ]
    ax[0].legend(handles=legend_elements, fontsize=legend_size)
    ax[0].set_ylabel(r"$P~[\AA]$", fontsize=fontsize)
    ax[1].set_ylabel("$P_{\mathrm{pipeline}} / P_{\mathrm{diff}}$", fontsize=fontsize)
    ax[1].set_xlabel("$z$", fontsize=fontsize)
    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(ratio_y_min, ratio_y_max)

    for i in [0, 1]:
        ax[i].xaxis.set_tick_params(labelsize=ticks_size)
        ax[i].yaxis.set_tick_params(labelsize=ticks_size)
        ax[i].margins(x=0)

    if out_name is not None:
        fig.savefig(f"{out_name}.pdf", format="pdf")
    if out_points is not None:
        np.savetxt(
            f"{out_name}.txt",
            np.transpose(
                np.stack(
                    [
                        mean_pk_k.zbin,
                        [mean_pk_k.p_noise[z] for z in mean_pk_k.zbin],
                        [mean_pk_k.err_noise[z] for z in mean_pk_k.zbin],
                        [mean_pk_k.p_diff[z] for z in mean_pk_k.zbin],
                        [mean_pk_k.err_diff[z] for z in mean_pk_k.zbin],
                        [mean_pk_k.noiseoverdiff[z] for z in mean_pk_k.zbin],
                        [mean_pk_k.err_noiseoverdiff[z] for z in mean_pk_k.zbin],
                    ]
                )
            ),
            header="REDSHIFT & NOISE POWER SPECTRA & STD NOISE POWER SPECTRA "
            + "& DIFF POWER SPECTRA & STD DIFF POWER SPECTRA "
            + "& RATIO POWER SPECTRA & STD RATIO POWER SPECTRA",
        )
