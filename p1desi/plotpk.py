#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""


import fitsio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from scipy.stats import binned_statistic

from p1desi import uncertainty, utils


def plot(
    pk,
    zmax,
    outname=None,
    outpoints=None,
    plot_P=False,
    systematics_file=None,
    pk_low_z=None,
    z_change=None,
    analysis_type="y1",
    **plot_args,
):
    zbins = pk.zbin[pk.zbin < zmax]

    marker_size = utils.return_key(plot_args, "marker_size", 7)
    marker_style = utils.return_key(plot_args, "marker_style", ".")
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.5)
    ymin = utils.return_key(plot_args, "ymin", 0.01)
    ymax = utils.return_key(plot_args, "ymax", 0.2)
    figsize = utils.return_key(plot_args, "figsize", (11, 8.5))
    place_velaxis = utils.return_key(plot_args, "place_velaxis", True)
    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        color = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    fig, ax = plt.subplots(1, figsize=figsize)

    if systematics_file is not None:
        if analysis_type == "edr":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics
        elif analysis_type == "y1":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics_y1
        (
            _,
            syste_tot,
            list_systematics,
            list_systematics_name,
        ) = function_uncertainty(systematics_file)

    zarr, karr, pkarr, errarr = [], [], [], []

    if systematics_file is not None:
        statarr, systarr = [], []
        systindivarr = [[] for i in range(len(list_systematics))]

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

        mask_k = (pk_to_plot.k[z] < kmax) & (pk_to_plot.k[z] > kmin)
        k = pk_to_plot.k[z][mask_k]

        list_systematics_z = []
        if plot_P:
            p = pk_to_plot.p[z][mask_k]
            stat = pk_to_plot.err[z][mask_k]

            if systematics_file is not None:
                syst = syste_tot[z][mask_k]
                for syst_indiv in list_systematics:
                    list_systematics_z.append(syst_indiv[z][mask_k])

        else:
            p = pk_to_plot.norm_p[z][mask_k]
            stat = pk_to_plot.norm_err[z][mask_k]
            if systematics_file is not None:
                syst = (pk_to_plot.k[z] * syste_tot[z] / np.pi)[mask_k]
                for syst_indiv in list_systematics:
                    list_systematics_z.append(
                        (pk_to_plot.k[z] * syst_indiv[z] / np.pi)[mask_k]
                    )

        if systematics_file is not None:
            error_bar = np.sqrt(stat**2 + syst**2)
        else:
            error_bar = stat

        zarr.append(np.array([z for j in range(len(k))]))
        karr.append(k)
        pkarr.append(p)
        errarr.append(error_bar)
        if systematics_file is not None:
            statarr.append(stat)
            systarr.append(syst)
            for j in range(len(list_systematics_z)):
                systindivarr[j].append(list_systematics_z[j])

        ax.errorbar(
            k,
            p,
            yerr=error_bar,
            fmt=marker_style,
            color=color[i],
            markersize=marker_size,
            label=r"$z = ${:1.1f}  ({} chunks)".format(z, pk_to_plot.number_chunks[z]),
        )

    ax.set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
    ax.set_ylabel(
        r"$\Delta_{1\mathrm{D},\alpha}^{2}$", fontsize=fontsize_y, labelpad=-1
    )

    ax.set_yscale("log")
    ax.xaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(direction="in")
    ax.yaxis.set_ticks_position("both")
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)

    ax.set_xlim(kmin, kmax)
    ax.set_ylim(ymin, ymax)

    ax.legend(loc=2, bbox_to_anchor=(1.03, 0.9), borderaxespad=0.0, fontsize=fontlegend)
    fig.subplots_adjust(
        top=0.75, bottom=0.08, left=0.08, right=0.7, hspace=0.2, wspace=0.2
    )
    if pk.velunits is False:
        if place_velaxis:
            utils.place_k_speed_unit_axis(
                fig, ax, fontsize=fontsize_x, size=labelsize, pos=0.15
            )

    if outname is not None:
        plt.tight_layout()
        fig.savefig(outname)

    if outpoints is not None:
        if systematics_file is not None:
            return_outpoints(
                outpoints,
                zarr,
                karr,
                pkarr,
                errarr,
                plot_P=plot_P,
                statarr=statarr,
                systarr=systarr,
                systindivarr=systindivarr,
                systindivname=list_systematics_name,
            )
        else:
            return_outpoints(outpoints, zarr, karr, pkarr, errarr, plot_P=plot_P)

    return (fig, ax)


def plot_comparison(
    pk,
    pk2,
    zmax,
    outname=None,
    outpoints=None,
    pk_low_z=None,
    z_change=None,
    plot_P=False,
    systematics_file=None,
    analysis_type="y1",
    **plot_args,
):
    if pk.velunits != pk2.velunits:
        raise ValueError(
            "The power spectrum you want to compare are expressed in different units"
        )

    zbins = pk.zbin[pk.zbin < zmax]

    marker_size = utils.return_key(plot_args, "marker_size", 7)
    marker_style = utils.return_key(plot_args, "marker_style", ".")
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.5)
    ymin = utils.return_key(plot_args, "ymin", 0.01)
    ymax = utils.return_key(plot_args, "ymax", 0.2)
    figsize = utils.return_key(plot_args, "figsize", (10, 8))
    place_velaxis = utils.return_key(plot_args, "place_velaxis", True)
    label = utils.return_key(plot_args, "label", "PK1")
    label2 = utils.return_key(plot_args, "label2", "PK2")
    alpha_comp = utils.return_key(plot_args, "alpha_comp", 0.4)
    marker_comp = utils.return_key(plot_args, "marker_comp", None)
    linestyle_comp = utils.return_key(plot_args, "linestyle_comp", None)
    extrapolate_ratio = utils.return_key(plot_args, "extrapolate_ratio", False)
    plot_error_ratio = utils.return_key(plot_args, "plot_error_ratio", False)
    plot_middle_error_ratio = utils.return_key(
        plot_args, "plot_middle_error_ratio", False
    )
    z_plot_middle_ratio_error = utils.return_key(
        plot_args, "z_plot_middle_ratio_error", 2.8
    )
    ymin_ratio = utils.return_key(plot_args, "ymin_ratio", 0.85)
    ymax_ratio = utils.return_key(plot_args, "ymax_ratio", 1.15)
    apply_mask_comp = utils.return_key(plot_args, "apply_mask_comp", True)
    zmax_comp = utils.return_key(plot_args, "zmax_comp", None)
    resample_pk = utils.return_key(plot_args, "resample_pk", False)
    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        color = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    fig, ax = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[3, 1]), sharex=True
    )

    if systematics_file is not None:
        if analysis_type == "edr":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics
        elif analysis_type == "y1":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics_y1
        (
            _,
            syste_tot,
            list_systematics,
            list_systematics_name,
        ) = function_uncertainty(systematics_file)

    zarr, karr, pkarr, errarr = [], [], [], []
    zarr2, karr2, pkarr2, errarr2 = [], [], [], []

    if systematics_file is not None:
        statarr, systarr = [], []
        systindivarr = [[] for i in range(len(list_systematics))]

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

        mask_k = (pk_to_plot.k[z] < kmax) & (pk_to_plot.k[z] > kmin)
        k = pk_to_plot.k[z][mask_k]

        if apply_mask_comp:
            mask_k2 = (pk2.k[z] < kmax) & (pk2.k[z] > kmin)
        else:
            mask_k2 = np.full(pk2.k[z].shape, True)
        k2 = pk2.k[z][mask_k2]
        list_systematics_z = []
        p_comp = pk_to_plot.p[z][mask_k]
        stat_comp = pk_to_plot.err[z][mask_k]
        p2_comp = pk2.p[z][mask_k2]
        error_bar2_comp = pk2.err[z][mask_k2]
        if systematics_file is not None:
            syst_comp = syste_tot[z][mask_k]
        if plot_P:
            p = pk_to_plot.p[z][mask_k]
            stat = pk_to_plot.err[z][mask_k]

            if systematics_file is not None:
                syst = syste_tot[z][mask_k]
                for syst_indiv in list_systematics:
                    list_systematics_z.append(syst_indiv[z][mask_k])

            p2 = pk2.p[z][mask_k2]
            error_bar2 = pk2.err[z][mask_k2]

        else:
            p = pk_to_plot.norm_p[z][mask_k]
            stat = pk_to_plot.norm_err[z][mask_k]
            if systematics_file is not None:
                syst = (pk_to_plot.k[z] * syste_tot[z] / np.pi)[mask_k]
                for syst_indiv in list_systematics:
                    list_systematics_z.append(
                        (pk_to_plot.k[z] * syst_indiv[z] / np.pi)[mask_k]
                    )

            p2 = pk2.norm_p[z][mask_k2]
            error_bar2 = pk2.norm_err[z][mask_k2]

        if resample_pk:
            k2_edges = np.zeros(len(k2) + 1)
            k2_edges[1:-1] = (k2[:-1] + k2[1:]) / 2
            k2_edges[0] = k2[0] - (k2[1] - k2[0]) / 2
            k2_edges[-1] = k2[-1] + ((k2[-1] - k2[-2]) / 2)

            number = binned_statistic(k, k, bins=k2_edges, statistic="count")

            means = binned_statistic(k, p_comp, bins=k2_edges, statistic="mean")
            p_comp = means.statistic

            means = binned_statistic(k, p, bins=k2_edges, statistic="mean")
            p = means.statistic

            means_stat = binned_statistic(k, stat_comp, bins=k2_edges, statistic="mean")
            stat_comp = means_stat.statistic / np.sqrt(number.statistic)

            means_stat = binned_statistic(k, stat, bins=k2_edges, statistic="mean")
            stat = means_stat.statistic / np.sqrt(number.statistic)

            if systematics_file is not None:
                means_syst = binned_statistic(k, syst, bins=k2_edges, statistic="mean")
                syst = means_syst.statistic / np.sqrt(number.statistic)

                means_syst = binned_statistic(
                    k, syst_comp, bins=k2_edges, statistic="mean"
                )
                syst_comp = means_syst.statistic / np.sqrt(number.statistic)

                for j in range(len(list_systematics_z)):
                    means_syst_indiv = binned_statistic(
                        k, list_systematics_z[j], bins=k2_edges, statistic="mean"
                    )
                    list_systematics_z[j] = means_syst_indiv.statistic / np.sqrt(
                        number.statistic
                    )

            k = (means.bin_edges[:-1] + means.bin_edges[1:]) / 2.0

        if systematics_file is not None:
            error_bar = np.sqrt(stat**2 + syst**2)
            error_bar_comp = np.sqrt(stat_comp**2 + syst_comp**2)
        else:
            error_bar = stat
            error_bar_comp = stat_comp

        zarr.append(np.array([z for j in range(len(k))]))
        karr.append(k)
        pkarr.append(p)
        errarr.append(error_bar)
        if systematics_file is not None:
            statarr.append(stat)
            systarr.append(syst)
            for j in range(len(list_systematics_z)):
                systindivarr[j].append(list_systematics_z[j])

        zarr2.append(np.array([z for j in range(len(k2))]))
        karr2.append(k2)
        pkarr2.append(p2)
        errarr2.append(error_bar2)

        ax[0].errorbar(
            k,
            p,
            yerr=error_bar,
            marker=marker_style,
            linestyle="None",
            color=color[i],
            markersize=marker_size,
        )
        if marker_comp is not None:
            ax[0].errorbar(
                k2,
                p2,
                error_bar2,
                marker=marker_comp,
                linestyle="None",
                color=color[i],
                markersize=marker_size,
            )
        elif linestyle_comp is not None:
            ax[0].plot(
                k2,
                p2,
                marker=None,
                linestyle=linestyle_comp,
                color=color[i],
                markersize=marker_size,
            )
        else:
            ax[0].fill_between(
                k2,
                p2 - error_bar2,
                p2 + error_bar2,
                color=color[i],
                alpha=alpha_comp,
            )

            ax[0].errorbar(
                k2,
                p2,
                error_bar2,
                marker="None",
                color=color[i],
                markersize=marker_size,
                alpha=alpha_comp,
            )
        if extrapolate_ratio:
            fill_value = "extrapolate"
        else:
            fill_value = np.nan
        p2_interp = interp1d(
            k2,
            p2_comp,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k)
        err_p2_interp = interp1d(
            k2,
            error_bar2_comp,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k)

        ratio = p_comp / p2_interp
        err_ratio = (p_comp / p2_interp) * np.sqrt(
            (error_bar_comp / p_comp) ** 2 + (err_p2_interp / p2_interp) ** 2
        )
        if zmax_comp is not None:
            if z > zmax_comp:
                plot_ratio = False
            else:
                plot_ratio = True

        else:
            plot_ratio = True
        if plot_ratio:
            if plot_error_ratio:
                ax[1].errorbar(
                    k,
                    ratio,
                    err_ratio,
                    marker=marker_style,
                    color=color[i],
                    markersize=marker_size,
                )
            else:
                ax[1].plot(
                    k,
                    ratio,
                    marker=marker_style,
                    color=color[i],
                    markersize=marker_size,
                )

        if z == z_plot_middle_ratio_error:
            err_ratio_plot = err_ratio
            k_ratio_plot = k

    if plot_middle_error_ratio:
        ax[1].fill_between(
            k_ratio_plot,
            1 - err_ratio_plot,
            1 + err_ratio_plot,
            color=f"k",
            alpha=0.3,
            hatch="///",
        )

    if pk.velunits:
        ax[1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
    else:
        ax[1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)

    ax[0].set_ylabel(
        r"$\Delta_{1\mathrm{D},\alpha}^{2}$", fontsize=fontsize_y, labelpad=-1
    )

    ax[0].set_yscale("log")
    ax[0].xaxis.set_ticks_position("both")
    ax[0].xaxis.set_tick_params(direction="in")
    ax[0].yaxis.set_ticks_position("both")
    ax[0].yaxis.set_tick_params(direction="in")
    ax[0].xaxis.set_tick_params(labelsize=labelsize)
    ax[0].yaxis.set_tick_params(labelsize=labelsize)

    ax[0].set_xlim(kmin, kmax)
    ax[0].set_ylim(ymin, ymax)
    ax[1].set_ylabel(f"{label}/{label2}", fontsize=fontsize_y)
    ax[1].set_ylim(ymin_ratio, ymax_ratio)
    ax[1].xaxis.set_tick_params(labelsize=labelsize)
    ax[1].yaxis.set_tick_params(labelsize=labelsize)

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

    legend_elements.append(
        Line2D(
            [],
            [],
            color="k",
            marker=marker_style,
            linestyle="None",
            label=label,
        )
    )
    if marker_comp is not None:
        legend_elements.append(
            Line2D(
                [],
                [],
                color="k",
                marker=marker_comp,
                linestyle="None",
                label=label2,
            )
        )
    elif linestyle_comp is not None:
        legend_elements.append(
            Line2D(
                [],
                [],
                color="k",
                marker=None,
                linestyle=linestyle_comp,
                label=label2,
            )
        )
    else:
        legend_elements.append(
            mpatches.Patch(color="k", alpha=alpha_comp, label=label2),
        )
    ax[0].legend(
        handles=legend_elements,
        loc=2,
        bbox_to_anchor=(1.03, 0.9),
        borderaxespad=0.0,
        fontsize=fontlegend,
    )
    if plot_middle_error_ratio:
        legend_elements = [
            Line2D([], [], color=f"k", marker=".", linestyle="-", label=f"Ratio"),
            mpatches.Patch(
                color="k",
                alpha=0.4,
                label=f"Error for \n z = {z_plot_middle_ratio_error}",
                hatch="///",
            ),
        ]

        ax[1].legend(
            handles=legend_elements,
            loc=2,
            bbox_to_anchor=(1.03, 0.9),
            borderaxespad=0.0,
            fontsize=fontlegend,
        )

    fig.subplots_adjust(
        top=0.75, bottom=0.08, left=0.08, right=0.7, hspace=0.2, wspace=0.2
    )
    if pk.velunits is False:
        if place_velaxis:
            utils.place_k_speed_unit_axis(
                fig, ax[0], fontsize=fontsize_x, size=labelsize, pos=0.15
            )

    if outname is not None:
        plt.tight_layout()
        fig.savefig(outname + ".pdf")
        fig.savefig(outname + ".png")

    if outpoints is not None:
        if systematics_file is not None:
            return_outpoints(
                outpoints,
                zarr,
                karr,
                pkarr,
                errarr,
                plot_P=plot_P,
                statarr=statarr,
                systarr=systarr,
                systindivarr=systindivarr,
                systindivname=list_systematics_name,
            )
        else:
            return_outpoints(outpoints, zarr, karr, pkarr, errarr, plot_P=plot_P)

        return_outpoints(outpoints + "_2", zarr2, karr2, pkarr2, errarr2, plot_P=plot_P)
    return (fig, ax)


def return_outpoints(
    outpoints,
    zarr,
    karr,
    pkarr,
    errarr,
    plot_P=False,
    statarr=None,
    systarr=None,
    systindivarr=None,
    systindivname=None,
):
    stack = []
    header = "Z & K & "
    if plot_P:
        header = header + "PK & "
        header = header + "ERR & "
    else:
        header = header + "K*PK/PI & "
        header = header + "K*ERR/PI & "
    stack.append(np.around(np.concatenate(zarr, axis=0), 3))
    stack.append(np.concatenate(karr, axis=0))
    stack.append(np.concatenate(pkarr, axis=0))
    stack.append(np.concatenate(errarr, axis=0))
    if statarr is not None:
        header = header + "STAT & SYST_TOT & "

        stack.append(np.concatenate(statarr, axis=0))
        stack.append(np.concatenate(systarr, axis=0))
        for i in range(len(systindivarr)):
            header = header + f"SYST_{systindivname[i]} & "
            stack.append(np.concatenate(systindivarr[i], axis=0))

    text_file = np.vstack(stack)
    np.savetxt(outpoints + ".txt", np.transpose(text_file), header=header)


def save_p1d(
    mean_pk,
    systematics_file,
    zmax,
    output_name,
    zedge_bin=0.2,
    blinding="desi_y1",
    analysis_type="y1",
    use_boot=True,
    verify_cov=True,
):

    zbin_save = mean_pk.zbin[mean_pk.zbin < zmax]

    kmin = mean_pk.kmin[zbin_save[0]]
    kmax = mean_pk.kmax[zbin_save[0]]
    n_k = mean_pk.k[zbin_save[0]].size
    k_edges = np.linspace(kmin, kmax, n_k + 1)
    k_edges_1 = k_edges[:-1]
    k_edges_2 = k_edges[1:]
    k_centers = (k_edges_1 + k_edges_2) / 2

    z_edges_1_full = np.concatenate(
        [np.full(mean_pk.k[z].shape, z - zedge_bin / 2) for z in zbin_save]
    )
    z_edges_2_full = np.concatenate(
        [np.full(mean_pk.k[z].shape, z + zedge_bin / 2) for z in zbin_save]
    )
    z = np.concatenate([np.full(mean_pk.k[z].shape, z) for z in zbin_save])

    k_edges_1_full = np.concatenate([k_edges_1 for z in zbin_save])
    k_edges_2_full = np.concatenate([k_edges_2 for z in zbin_save])
    k_centers_full = np.concatenate([k_centers for z in zbin_save])
    pk = np.concatenate([mean_pk.p[z] for z in zbin_save])
    error_stat = np.concatenate([mean_pk.err[z] for z in zbin_save])
    p_raw = np.concatenate([mean_pk.p_raw[z] for z in zbin_save])
    p_noise = np.concatenate([mean_pk.p_noise[z] for z in zbin_save])

    if use_boot:
        cov = block_diag(*[mean_pk.boot_cov[z].reshape(n_k, n_k) for z in zbin_save])
    else:
        cov = block_diag(*[mean_pk.cov[z].reshape(n_k, n_k) for z in zbin_save])

    if analysis_type == "edr":
        function_uncertainty = uncertainty.prepare_uncertainty_systematics
    elif analysis_type == "y1":
        function_uncertainty = uncertainty.prepare_uncertainty_systematics_y1
    (
        _,
        syste_tot,
        list_systematics,
        list_systematics_name,
    ) = function_uncertainty(systematics_file)

    cov_syst = np.zeros_like(cov)
    for i in range(len(list_systematics_name)):
        cov_syst = cov_syst + block_diag(
            *[
                np.outer(list_systematics[i][z], list_systematics[i][z])
                for z in zbin_save
            ]
        )

    full_cov = cov + cov_syst

    error_syst = np.concatenate([syste_tot[z] for z in zbin_save])
    error_pk = np.sqrt(error_stat**2 + error_syst**2)

    fits = fitsio.FITS(output_name, "rw", clobber=True)

    dtype = [
        ("Z1", "f8"),
        ("Z2", "f8"),
        ("Z", "f8"),
        ("K1", "f8"),
        ("K2", "f8"),
        ("K", "f8"),
        ("PLYA", "f8"),
        ("E_PK", "f8"),
        ("E_STAT", "f8"),
        ("E_SYST", "f8"),
        ("PRAW", "f8"),
        ("PNOISE", "f8"),
    ]

    if mean_pk.velunits:
        units = [
            "",
            "",
            "",
            "(km/s)^(-1)",
            "(km/s)^(-1)",
            "(km/s)^(-1)",
            "(km/s)",
            "(km/s)",
            "(km/s)",
            "(km/s)",
            "(km/s)",
            "(km/s)",
        ]
    else:
        units = [
            "",
            "",
            "",
            "AA^(-1)",
            "AA^(-1)",
            "AA^(-1)",
            "AA",
            "AA",
            "AA",
            "AA",
            "AA",
            "AA",
        ]

    hdu = np.zeros(z.size, dtype=dtype)
    hdu["Z1"] = z_edges_1_full
    hdu["Z2"] = z_edges_2_full
    hdu["Z"] = z
    hdu["K1"] = k_edges_1_full
    hdu["K2"] = k_edges_2_full
    hdu["K"] = k_centers_full
    hdu["PLYA"] = pk
    hdu["E_PK"] = error_pk
    hdu["E_STAT"] = error_stat
    hdu["E_SYST"] = error_syst
    hdu["PRAW"] = p_raw
    hdu["PNOISE"] = p_noise

    header = {
        "ZMIN": np.min(zbin_save),
        "ZMAX": np.max(zbin_save),
        "NZ": len(zbin_save),
        "KMIN": kmin,
        "KMAX": kmax,
        "NK": n_k,
        "VELUNITS": mean_pk.velunits,
        "BLINDING": blinding,
    }
    if blinding is not None:
        fits.write(hdu, header=header, units=units, extname="P1D_BLIND")
    else:
        fits.write(hdu, header=header, units=units, extname="P1D")

    dtype_systematics = [
        ("Z", "f8"),
        ("K", "f8"),
        ("E_SYST", "f8"),
    ]
    if mean_pk.velunits:
        units_systematics = [
            "",
            "(km/s)^(-1)",
            "(km/s)",
        ]
    else:
        units_systematics = [
            "",
            "AA^(-1)",
            "AA",
        ]

    for i in range(len(list_systematics_name)):
        name_syste = "E_" + list_systematics_name[i].upper().replace(" ", "_")
        if name_syste == "DLA":
            name_syste = "DLA_MASKING"
        dtype_systematics.append((name_syste, "f8"))
        if mean_pk.velunits:
            units_systematics.append("(km/s)")
        else:
            units_systematics.append("AA")

    hdu_systematics = np.zeros(z.size, dtype=dtype_systematics)
    hdu_systematics["Z"] = z
    hdu_systematics["K"] = k_centers_full
    hdu_systematics["E_SYST"] = error_syst

    for i in range(len(list_systematics_name)):
        name_syste = "E_" + list_systematics_name[i].upper().replace(" ", "_")
        if name_syste == "DLA":
            name_syste = "DLA_MASKING"
        hdu_systematics[name_syste] = np.concatenate(
            [list_systematics[i][z] for z in zbin_save]
        )

    header = {
        "ZMIN": np.min(zbin_save),
        "ZMAX": np.max(zbin_save),
        "NZ": len(zbin_save),
        "KMIN": kmin,
        "KMAX": kmax,
        "NK": n_k,
        "VELUNITS": mean_pk.velunits,
    }
    fits.write(
        hdu_systematics, header=header, units=units_systematics, extname="SYSTEMATICS"
    )

    header = {
        "ZMIN": np.min(zbin_save),
        "ZMAX": np.max(zbin_save),
        "NZ": len(zbin_save),
        "KMIN": kmin,
        "KMAX": kmax,
        "NK": n_k,
        "VELUNITS": mean_pk.velunits,
        "IS_BD": False,
    }
    if mean_pk.velunits:
        units = ["(km/s)^2"]
    else:
        units = ["AA^2"]

    if verify_cov:
        eig_full = np.linalg.eigvals(full_cov)
        if len([eig_full[eig_full < 0.0]]):
            print(
                "Full covariance matrix has negative eigenvalues",
                "First ten values:",
                eig_full[eig_full < 0.0][:10],
            )
        eig = np.linalg.eigvals(cov)
        if len([eig[eig < 0.0]]):
            print(
                "Statistical covariance matrix has negative eigenvalues",
                "First ten values:",
                eig[eig < 0.0][:10],
            )
        eig_syst = np.linalg.eigvals(cov_syst)
        if len([eig_syst[eig_syst < 0.0]]):
            print(
                "Systematics covariance matrix has negative eigenvalues",
                "First ten values:",
                eig_syst[eig_syst < 0.0][:10],
            )

    fits.write(full_cov, header=header, units=units, extname="COVARIANCE")
    fits.write(cov, header=header, units=units, extname="COVARIANCE_STAT")
    fits.write(cov_syst, header=header, units=units, extname="COVARIANCE_SYST")

    fits.close()


def plot_three_comparison(
    pk,
    pk2,
    pk3,
    zmax,
    outname=None,
    outpoints=None,
    plot_P=False,
    systematics_file=None,
    analysis_type="y1",
    **plot_args,
):
    if pk.velunits != pk2.velunits:
        raise ValueError(
            "The power spectrum you want to compare are expressed in different units"
        )

    zbins = pk.zbin[pk.zbin < zmax]

    marker_size = utils.return_key(plot_args, "marker_size", 7)
    marker_style = utils.return_key(plot_args, "marker_style", ".")
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.5)
    ymin = utils.return_key(plot_args, "ymin", 0.01)
    ymax = utils.return_key(plot_args, "ymax", 0.2)
    figsize = utils.return_key(plot_args, "figsize", (10, 8))
    place_velaxis = utils.return_key(plot_args, "place_velaxis", True)
    label = utils.return_key(plot_args, "label", "PK1")
    label2 = utils.return_key(plot_args, "label2", "PK2")
    label3 = utils.return_key(plot_args, "label3", "PK3 CORR")
    alpha_comp = utils.return_key(plot_args, "alpha_comp", 0.4)
    marker_comp = utils.return_key(plot_args, "marker_comp", None)
    linestyle_comp = utils.return_key(plot_args, "linestyle_comp", None)
    extrapolate_ratio = utils.return_key(plot_args, "extrapolate_ratio", False)
    plot_error_ratio = utils.return_key(plot_args, "plot_error_ratio", False)
    plot_middle_error_ratio = utils.return_key(
        plot_args, "plot_middle_error_ratio", False
    )
    z_plot_middle_ratio_error = utils.return_key(
        plot_args, "z_plot_middle_ratio_error", 2.8
    )
    ymin_ratio = utils.return_key(plot_args, "ymin_ratio", 0.85)
    ymax_ratio = utils.return_key(plot_args, "ymax_ratio", 1.15)
    apply_mask_comp = utils.return_key(plot_args, "apply_mask_comp", True)
    zmax_comp = utils.return_key(plot_args, "zmax_comp", None)
    resample_pk = utils.return_key(plot_args, "resample_pk", False)
    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        color = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    fig, ax = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[3, 1, 1]), sharex=True
    )

    if systematics_file is not None:
        if analysis_type == "edr":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics
        elif analysis_type == "y1":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics_y1
        (
            _,
            syste_tot,
            list_systematics,
            list_systematics_name,
        ) = function_uncertainty(systematics_file)

    zarr, karr, pkarr, errarr = [], [], [], []
    zarr2, karr2, pkarr2, errarr2 = [], [], [], []
    zarr3, karr3, pkarr3, errarr3 = [], [], [], []

    if systematics_file is not None:
        statarr, systarr = [], []
        systindivarr = [[] for i in range(len(list_systematics))]

    for i, z in enumerate(zbins):
        if pk.velunits:
            kmax = float(utils.kAAtokskm(kmax_AA, z=z))
            kmin = float(utils.kAAtokskm(kmin_AA, z=z))
        else:
            kmax = kmax_AA
            kmin = kmin_AA

        mask_k = (pk.k[z] < kmax) & (pk.k[z] > kmin)
        k = pk.k[z][mask_k]

        if apply_mask_comp:
            mask_k2 = (pk2.k[z] < kmax) & (pk2.k[z] > kmin)
            mask_k3 = (pk3.k[z] < kmax) & (pk3.k[z] > kmin)
        else:
            mask_k2 = np.full(pk2.k[z].shape, True)
            mask_k3 = np.full(pk3.k[z].shape, True)

        k2 = pk2.k[z][mask_k2]
        k3 = pk3.k[z][mask_k3]
        list_systematics_z = []
        if plot_P:
            p = pk.p[z][mask_k]
            stat = pk.err[z][mask_k]

            if systematics_file is not None:
                syst = syste_tot[z][mask_k]
                for syst_indiv in list_systematics:
                    list_systematics_z.append(syst_indiv[z][mask_k])

            p2 = pk2.p[z][mask_k2]
            error_bar2 = pk2.err[z][mask_k2]

            p3 = pk3.p[z][mask_k3]
            error_bar3 = pk3.err[z][mask_k3]

        else:
            p = pk.norm_p[z][mask_k]
            stat = pk.norm_err[z][mask_k]
            if systematics_file is not None:
                syst = (pk.k[z] * syste_tot[z] / np.pi)[mask_k]
                for syst_indiv in list_systematics:
                    list_systematics_z.append((pk.k[z] * syst_indiv[z] / np.pi)[mask_k])

            p2 = pk2.norm_p[z][mask_k2]
            error_bar2 = pk2.norm_err[z][mask_k2]
            p3 = pk3.norm_p[z][mask_k3]
            error_bar3 = pk3.norm_err[z][mask_k3]

        if resample_pk:
            k2_edges = np.zeros(len(k2) + 1)
            k2_edges[1:-1] = (k2[:-1] + k2[1:]) / 2
            k2_edges[0] = k2[0] - (k2[1] - k2[0]) / 2
            k2_edges[-1] = k2[-1] + ((k2[-1] - k2[-2]) / 2)

            number = binned_statistic(k, k, bins=k2_edges, statistic="count")

            means = binned_statistic(k, p, bins=k2_edges, statistic="mean")
            p = means.statistic

            means_stat = binned_statistic(k, stat, bins=k2_edges, statistic="mean")
            stat = means_stat.statistic / np.sqrt(number.statistic)

            if systematics_file is not None:
                means_syst = binned_statistic(k, syst, bins=k2_edges, statistic="mean")
                syst = means_syst.statistic / np.sqrt(number.statistic)

                for j in range(len(list_systematics_z)):
                    means_syst_indiv = binned_statistic(
                        k, list_systematics_z[j], bins=k2_edges, statistic="mean"
                    )
                    list_systematics_z[j] = means_syst_indiv.statistic / np.sqrt(
                        number.statistic
                    )

            k = (means.bin_edges[:-1] + means.bin_edges[1:]) / 2.0

        if systematics_file is not None:
            error_bar = np.sqrt(stat**2 + syst**2)
        else:
            error_bar = stat

        zarr.append(np.array([z for j in range(len(k))]))
        karr.append(k)
        pkarr.append(p)
        errarr.append(error_bar)
        if systematics_file is not None:
            statarr.append(stat)
            systarr.append(syst)
            for j in range(len(list_systematics_z)):
                systindivarr[j].append(list_systematics_z[j])

        zarr2.append(np.array([z for j in range(len(k2))]))
        karr2.append(k2)
        pkarr2.append(p2)
        errarr2.append(error_bar2)

        zarr3.append(np.array([z for j in range(len(k3))]))
        karr3.append(k3)
        pkarr3.append(p3)
        errarr3.append(error_bar3)

        ax[0].errorbar(
            k,
            p,
            yerr=error_bar,
            marker=marker_style,
            linestyle="None",
            color=color[i],
            markersize=marker_size,
        )
        if marker_comp is not None:
            ax[0].errorbar(
                k2,
                p2,
                error_bar2,
                marker=marker_comp,
                linestyle="None",
                color=color[i],
                markersize=marker_size,
            )
        elif linestyle_comp is not None:
            ax[0].plot(
                k2,
                p2,
                marker=None,
                linestyle=linestyle_comp,
                color=color[i],
                markersize=marker_size,
            )
        else:
            ax[0].fill_between(
                k2,
                p2 - error_bar2,
                p2 + error_bar2,
                color=color[i],
                alpha=alpha_comp,
            )

            ax[0].errorbar(
                k2,
                p2,
                error_bar2,
                marker="None",
                color=color[i],
                markersize=marker_size,
                alpha=alpha_comp,
            )

        ax[0].errorbar(
            k3,
            p3,
            error_bar3,
            linestyle="--",
            color=color[i],
        )

        if extrapolate_ratio:
            fill_value = "extrapolate"
        else:
            fill_value = np.nan
        p2_interp = interp1d(
            k2,
            p2,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k)
        err_p2_interp = interp1d(
            k2,
            error_bar2,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k)

        p3_interp = interp1d(
            k3,
            p3,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k2)
        err_p3_interp = interp1d(
            k3,
            error_bar3,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k2)

        ratio = p / p2_interp
        err_ratio = (p / p2_interp) * np.sqrt(
            (error_bar / p) ** 2 + (err_p2_interp / p2_interp) ** 2
        )

        ratio2 = p2 / p3_interp
        err_ratio2 = (p2 / p3_interp) * np.sqrt(
            (error_bar2 / p2) ** 2 + (err_p3_interp / p3_interp) ** 2
        )

        if zmax_comp is not None:
            if z > zmax_comp:
                plot_ratio = False
            else:
                plot_ratio = True

        else:
            plot_ratio = True
        if plot_ratio:
            if plot_error_ratio:
                ax[1].errorbar(
                    k,
                    ratio,
                    err_ratio,
                    marker=marker_style,
                    color=color[i],
                    markersize=marker_size,
                )
            else:
                ax[1].plot(
                    k,
                    ratio,
                    marker=marker_style,
                    color=color[i],
                    markersize=marker_size,
                )
            ax[2].plot(
                k2,
                ratio2,
                marker=marker_style,
                color=color[i],
                markersize=marker_size,
            )
        if z == z_plot_middle_ratio_error:
            err_ratio_plot = err_ratio
            k_ratio_plot = k

    if plot_middle_error_ratio:
        ax[1].fill_between(
            k_ratio_plot,
            1 - err_ratio_plot,
            1 + err_ratio_plot,
            color=f"k",
            alpha=0.3,
            hatch="///",
        )
        ax[2].fill_between(
            k_ratio_plot,
            1 - err_ratio_plot,
            1 + err_ratio_plot,
            color=f"k",
            alpha=0.3,
            hatch="///",
        )

    if pk.velunits:
        ax[2].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
    else:
        ax[2].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)

    ax[0].set_ylabel(
        r"$\Delta_{1\mathrm{D},\alpha}^{2}$", fontsize=fontsize_y, labelpad=-1
    )

    ax[0].set_yscale("log")
    ax[0].xaxis.set_ticks_position("both")
    ax[0].xaxis.set_tick_params(direction="in")
    ax[0].yaxis.set_ticks_position("both")
    ax[0].yaxis.set_tick_params(direction="in")
    ax[0].xaxis.set_tick_params(labelsize=labelsize)
    ax[0].yaxis.set_tick_params(labelsize=labelsize)

    ax[0].set_xlim(kmin, kmax)
    ax[0].set_ylim(ymin, ymax)
    ax[1].set_ylabel(f"{label}/{label2}", fontsize=fontsize_y)
    ax[1].set_ylim(ymin_ratio, ymax_ratio)
    ax[1].xaxis.set_tick_params(labelsize=labelsize)
    ax[1].yaxis.set_tick_params(labelsize=labelsize)

    ax[2].set_ylabel(f"{label2}/{label3}", fontsize=fontsize_y)
    ax[2].set_ylim(ymin_ratio, ymax_ratio)
    ax[2].xaxis.set_tick_params(labelsize=labelsize)
    ax[2].yaxis.set_tick_params(labelsize=labelsize)

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

    legend_elements.append(
        Line2D(
            [],
            [],
            color="k",
            marker=marker_style,
            linestyle="None",
            label=label,
        )
    )
    if marker_comp is not None:
        legend_elements.append(
            Line2D(
                [],
                [],
                color="k",
                marker=marker_comp,
                linestyle="None",
                label=label2,
            )
        )
    elif linestyle_comp is not None:
        legend_elements.append(
            Line2D(
                [],
                [],
                color="k",
                marker=None,
                linestyle=linestyle_comp,
                label=label2,
            )
        )
    else:
        legend_elements.append(
            mpatches.Patch(color="k", alpha=alpha_comp, label=label2),
        )
    legend_elements.append(
        Line2D(
            [],
            [],
            color="k",
            linestyle="--",
            label=label3,
        )
    )

    ax[0].legend(
        handles=legend_elements,
        loc=2,
        bbox_to_anchor=(1.03, 0.9),
        borderaxespad=0.0,
        fontsize=fontlegend,
    )
    if plot_middle_error_ratio:
        legend_elements = [
            Line2D([], [], color=f"k", marker=".", linestyle="-", label=f"Ratio"),
            mpatches.Patch(
                color="k",
                alpha=0.4,
                label=f"Error for \n z = {z_plot_middle_ratio_error}",
                hatch="///",
            ),
        ]

        ax[1].legend(
            handles=legend_elements,
            loc=2,
            bbox_to_anchor=(1.03, 0.9),
            borderaxespad=0.0,
            fontsize=fontlegend,
        )

    fig.subplots_adjust(
        top=0.75, bottom=0.08, left=0.08, right=0.7, hspace=0.2, wspace=0.2
    )
    if pk.velunits is False:
        if place_velaxis:
            utils.place_k_speed_unit_axis(
                fig, ax[0], fontsize=fontsize_x, size=labelsize, pos=0.15
            )

    if outname is not None:
        plt.tight_layout()
        fig.savefig(outname + ".pdf")
        fig.savefig(outname + ".png")

    if outpoints is not None:
        if systematics_file is not None:
            return_outpoints(
                outpoints,
                zarr,
                karr,
                pkarr,
                errarr,
                plot_P=plot_P,
                statarr=statarr,
                systarr=systarr,
                systindivarr=systindivarr,
                systindivname=list_systematics_name,
            )
        else:
            return_outpoints(outpoints, zarr, karr, pkarr, errarr, plot_P=plot_P)

        return_outpoints(outpoints + "_2", zarr2, karr2, pkarr2, errarr2, plot_P=plot_P)
        return_outpoints(outpoints + "_3", zarr3, karr3, pkarr3, errarr3, plot_P=plot_P)
    return (fig, ax)


def plot_variations(
    mean_pk_ref,
    mean_pk_list_variation,
    label_variations,
    plot_P,
    zmax,
    outname=None,
    outpoints=None,
    **plot_args,
):

    zbins = mean_pk_ref.zbin[mean_pk_ref.zbin < zmax]

    marker_size = utils.return_key(plot_args, "marker_size", 7)
    marker_style = utils.return_key(plot_args, "marker_style", ".")
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    label_base = utils.return_key(plot_args, "label_base", "BASE")
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.0)
    ymin = utils.return_key(plot_args, "ymin", None)
    ymax = utils.return_key(plot_args, "ymax", None)
    figsize = utils.return_key(plot_args, "figsize", (10, 8))
    extrapolate_ratio = utils.return_key(plot_args, "extrapolate_ratio", False)
    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        color = [f"C{i}" for i, z in enumerate(mean_pk_ref.zbin) if z < zmax]
    elif color_map == "rainbow":
        color = cm.rainbow(
            np.linspace(0, 1, len(mean_pk_ref.zbin[mean_pk_ref.zbin < zmax]))
        )

    fig, ax = plt.subplots(len(mean_pk_list_variation), 1, figsize=figsize)
    if len(mean_pk_list_variation) == 1:
        ax = [ax]

    z_arr, k_arr, ratio_arr, err_arr = (
        [],
        [],
        [[] for j in range(len(mean_pk_list_variation))],
        [[] for j in range(len(mean_pk_list_variation))],
    )
    for i, z in enumerate(zbins):
        if mean_pk_ref.velunits:
            kmax = float(utils.kAAtokskm(kmax_AA, z=z))
            kmin = float(utils.kAAtokskm(kmin_AA, z=z))
        else:
            kmax = kmax_AA
            kmin = kmin_AA

        mask_k = (mean_pk_ref.k[z] < kmax) & (mean_pk_ref.k[z] > kmin)
        k = mean_pk_ref.k[z][mask_k]

        z_arr.append([z for i in range(len(k))])
        k_arr.append(k)
        for j, mean_pk_comp in enumerate(mean_pk_list_variation):
            mask_k2 = (mean_pk_comp.k[z] < kmax) & (mean_pk_comp.k[z] > kmin)
            k2 = mean_pk_comp.k[z][mask_k2]
            if plot_P:
                p = mean_pk_ref.p[z][mask_k]
                error_bar = mean_pk_ref.err[z][mask_k]
                p2 = mean_pk_comp.p[z][mask_k2]
                error_bar2 = mean_pk_comp.err[z][mask_k2]

            else:
                p = mean_pk_ref.norm_p[z][mask_k]
                error_bar = mean_pk_ref.norm_err[z][mask_k]
                p2 = mean_pk_comp.norm_p[z][mask_k2]
                error_bar2 = mean_pk_comp.norm_err[z][mask_k2]

            if extrapolate_ratio:
                fill_value = "extrapolate"
            else:
                fill_value = np.nan
            p2_interp = interp1d(
                k2,
                p2,
                kind="linear",
                bounds_error=False,
                fill_value=fill_value,
            )(k)
            err_p2_interp = interp1d(
                k2,
                error_bar2,
                kind="linear",
                bounds_error=False,
                fill_value=fill_value,
            )(k)

            ratio = p / p2_interp
            err_ratio = (p / p2_interp) * np.sqrt(
                (error_bar / p) ** 2 + (err_p2_interp / p2_interp) ** 2
            )
            ratio_arr[j].append(ratio)
            err_arr[j].append(err_ratio)
            ax[j].errorbar(
                k,
                ratio,
                err_ratio,
                marker=marker_style,
                color=color[i],
                markersize=marker_size,
                label=r"$z = ${:1.1f}".format(z),
            )

    (
        z_arr,
        k_arr,
    ) = np.concatenate(
        z_arr
    ), np.concatenate(k_arr)

    for j in range(len(ratio_arr)):
        ratio_arr[j], err_arr[j] = np.concatenate(ratio_arr[j]), np.concatenate(
            err_arr[j]
        )

    for j, label in enumerate(label_variations):
        ax[j].set_ylabel(f"{label_base}/{label}", fontsize=fontsize_y)
        ax[j].xaxis.set_tick_params(labelsize=labelsize)
        ax[j].yaxis.set_tick_params(labelsize=labelsize)
        ax[j].set_ylim(bottom=ymin, top=ymax)
    if mean_pk_ref.velunits:
        ax[-1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
    else:
        ax[-1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)
    ax[0].legend(ncol=4, fontsize=fontlegend)
    fig.tight_layout()

    if outname is not None:
        fig.savefig(f"{outname}.pdf")
        fig.savefig(f"{outname}.png")
    if outpoints is not None:
        stack_array = [z_arr, k_arr]
        for j in range(len(ratio_arr)):
            stack_array.append(ratio_arr[j])
            stack_array.append(err_arr[j])
        if mean_pk_ref.velunits:
            header = "REDSHIFT & WAVENUMBER [s.km^-1]"
        else:
            header = "REDSHIFT & WAVENUMBER [Ang^-1]"
        for label in label_variations:
            header = header + " & RATIO BASE/{label} & ERROR RATIO BASE/{label}"
        np.savetxt(
            f"{outpoints}.txt",
            np.transpose(np.stack(stack_array)),
            header=header,
        )


def plot_three_comparison_bis(
    pk,
    pk2,
    pk3,
    zmax,
    outname=None,
    outpoints=None,
    pk_low_z=None,
    z_change=None,
    systematics_file=None,
    analysis_type="y1",
    **plot_args,
):

    zbins = pk.zbin[pk.zbin < zmax]

    marker_size = utils.return_key(plot_args, "marker_size", 7)
    marker_style = utils.return_key(plot_args, "marker_style", ".")
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.5)
    ymin = utils.return_key(plot_args, "ymin", 0.85)
    ymax = utils.return_key(plot_args, "ymax", 1.15)
    figsize = utils.return_key(plot_args, "figsize", (10, 8))
    label = utils.return_key(plot_args, "label", "PK1")
    label2 = utils.return_key(plot_args, "label2", "PK2")
    label3 = utils.return_key(plot_args, "label3", "PK3")
    extrapolate_ratio = utils.return_key(plot_args, "extrapolate_ratio", False)
    plot_error_ratio = utils.return_key(plot_args, "plot_error_ratio", False)
    plot_middle_error_ratio = utils.return_key(
        plot_args, "plot_middle_error_ratio", False
    )
    z_plot_middle_ratio_error = utils.return_key(
        plot_args, "z_plot_middle_ratio_error", 2.8
    )
    apply_mask_comp = utils.return_key(plot_args, "apply_mask_comp", True)
    resample_pk = utils.return_key(plot_args, "resample_pk", False)
    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        color = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        color = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if systematics_file is not None:
        if analysis_type == "edr":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics
        elif analysis_type == "y1":
            function_uncertainty = uncertainty.prepare_uncertainty_systematics_y1
        (
            _,
            syste_tot,
            _,
            _,
        ) = function_uncertainty(systematics_file)

    zarr, karr, ratio_arr = [], [], []
    zarr2, karr2, ratio_arr2 = [], [], []

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

        mask_k = (pk_to_plot.k[z] < kmax) & (pk_to_plot.k[z] > kmin)
        k = pk_to_plot.k[z][mask_k]

        if apply_mask_comp:
            mask_k2 = (pk2.k[z] < kmax) & (pk2.k[z] > kmin)
        else:
            mask_k2 = np.full(pk2.k[z].shape, True)
        k2 = pk2.k[z][mask_k2]

        if apply_mask_comp:
            mask_k3 = (pk3.k[z] < kmax) & (pk3.k[z] > kmin)
        else:
            mask_k3 = np.full(pk3.k[z].shape, True)
        k3 = pk3.k[z][mask_k3]

        p = pk_to_plot.p[z][mask_k]
        stat = pk_to_plot.err[z][mask_k]

        if systematics_file is not None:
            syst = syste_tot[z][mask_k]

        p2 = pk2.p[z][mask_k2]
        error_bar2 = pk2.err[z][mask_k2]
        p3 = pk3.p[z][mask_k3]
        error_bar3 = pk3.err[z][mask_k3]

        if resample_pk:
            k3_edges = np.zeros(len(k3) + 1)
            k3_edges[1:-1] = (k3[:-1] + k3[1:]) / 2
            k3_edges[0] = k3[0] - (k3[1] - k3[0]) / 2
            k3_edges[-1] = k3[-1] + ((k3[-1] - k3[-2]) / 2)

            number = binned_statistic(k, k, bins=k3_edges, statistic="count")

            means = binned_statistic(k, p, bins=k3_edges, statistic="mean")
            p_resample = means.statistic

            means_stat = binned_statistic(k, stat, bins=k3_edges, statistic="mean")
            stat_resample = means_stat.statistic / np.sqrt(number.statistic)

            if systematics_file is not None:
                means_syst = binned_statistic(k, syst, bins=k3_edges, statistic="mean")
                syst_resample = means_syst.statistic / np.sqrt(number.statistic)
                error_bar_resample = np.sqrt(stat_resample**2 + syst_resample**2)
            else:
                error_bar_resample = stat_resample**2

            k_resample = (means.bin_edges[:-1] + means.bin_edges[1:]) / 2.0

        if systematics_file is not None:
            error_bar = np.sqrt(stat**2 + syst**2)
        else:
            error_bar = stat

        if extrapolate_ratio:
            fill_value = "extrapolate"
        else:
            fill_value = np.nan
        p2_interp = interp1d(
            k2,
            p2,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k)
        err_p2_interp = interp1d(
            k2,
            error_bar2,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k)
        if resample_pk:
            k_interp = k_resample
            p_interp = p_resample
        else:
            k_interp = k
            p_interp = p

        p3_interp = interp1d(
            k3,
            p3,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k_interp)
        err_p3_interp = interp1d(
            k3,
            error_bar3,
            kind="linear",
            bounds_error=False,
            fill_value=fill_value,
        )(k_interp)

        ratio = p / p2_interp
        err_ratio = (p / p2_interp) * np.sqrt(
            (error_bar / p) ** 2 + (err_p2_interp / p2_interp) ** 2
        )

        ratio2 = p_interp / p3_interp
        err_ratio2 = (p_interp / p3_interp) * np.sqrt(
            (error_bar_resample / p_interp) ** 2 + (err_p3_interp / p3_interp) ** 2
        )
        zarr.append(np.array([z for j in range(len(k))]))
        karr.append(k)

        zarr2.append(np.array([z for j in range(len(k_interp))]))
        karr2.append(k_interp)

        ratio_arr.append(ratio)
        ratio_arr2.append(ratio2)

        if plot_error_ratio:
            ax[0].errorbar(
                k,
                ratio,
                err_ratio,
                marker=marker_style,
                color=color[i],
                markersize=marker_size,
            )
            ax[1].errorbar(
                k_interp,
                ratio2,
                err_ratio2,
                marker=marker_style,
                color=color[i],
                markersize=marker_size,
            )
        else:
            ax[0].plot(
                k,
                ratio,
                marker=marker_style,
                color=color[i],
                markersize=marker_size,
            )
            ax[1].plot(
                k_interp,
                ratio2,
                marker=marker_style,
                color=color[i],
                markersize=marker_size,
            )
        if z == z_plot_middle_ratio_error:
            err_ratio_plot = err_ratio
            k_ratio_plot = k
            err_ratio_plot2 = err_ratio2
            k_ratio_plot2 = k_interp

    if plot_middle_error_ratio:
        ax[0].fill_between(
            k_ratio_plot,
            1 - err_ratio_plot,
            1 + err_ratio_plot,
            color=f"k",
            alpha=0.3,
            hatch="///",
        )
        ax[1].fill_between(
            k_ratio_plot2,
            1 - err_ratio_plot2,
            1 + err_ratio_plot2,
            color=f"k",
            alpha=0.3,
            hatch="///",
        )
    if pk.velunits:
        ax[1].set_xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$", fontsize=fontsize_x
        )
    else:
        ax[1].set_xlabel(r"$k~[\mathrm{\AA}^{-1}]$", fontsize=fontsize_x)

    ax[0].set_ylabel(
        r"$\Delta_{1\mathrm{D},\alpha}^{2}$", fontsize=fontsize_y, labelpad=-1
    )

    ax[0].xaxis.set_tick_params(labelsize=labelsize)
    ax[0].yaxis.set_tick_params(labelsize=labelsize)
    ax[0].set_xlim(kmin, kmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_ylabel(f"{label}/{label2}", fontsize=fontsize_y)

    ax[1].set_xlim(kmin, kmax)
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_ylabel(f"{label}/{label3}", fontsize=fontsize_y)
    ax[1].xaxis.set_tick_params(labelsize=labelsize)
    ax[1].yaxis.set_tick_params(labelsize=labelsize)

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

    if plot_middle_error_ratio:
        legend_elements.append(
            mpatches.Patch(
                color="k",
                alpha=0.4,
                label=f"Error for \n z = {z_plot_middle_ratio_error}",
                hatch="///",
            ),
        )

    ax[0].legend(
        handles=legend_elements,
        loc=2,
        bbox_to_anchor=(1.03, 1.0),
        borderaxespad=0.0,
        fontsize=fontlegend,
    )

    plt.tight_layout()
    if outname is not None:
        fig.savefig(outname + ".pdf")
        fig.savefig(outname + ".png")

    if outpoints is not None:
        zarr = np.concatenate(zarr, axis=0)
        karr = np.concatenate(karr, axis=0)
        zarr2 = np.concatenate(zarr2, axis=0)
        karr2 = np.concatenate(karr2, axis=0)
        ratio_arr = np.concatenate(ratio_arr, axis=0)
        ratio_arr2 = np.concatenate(ratio_arr2, axis=0)

        if pk.velunits:
            header = "REDSHIFT & WAVENUMBER 1 [s.km^-1] & RATIO"
        else:
            header = "REDSHIFT & WAVENUMBER 1 [Ang^-1] & RATIO"
        np.savetxt(
            f"{outpoints}.txt",
            np.transpose(np.stack([zarr, karr, ratio_arr])),
            header=header,
        )
        np.savetxt(
            f"{outpoints}_2.txt",
            np.transpose(np.stack([zarr2, karr2, ratio_arr2])),
            header=header,
        )
    return (fig, ax)


def print_data_numbers(
    mean_pk_ref,
    mean_pk_list_variation,
    zmax,
):
    zbins = mean_pk_ref.zbin[mean_pk_ref.zbin < zmax]
    N_ref = [mean_pk_ref.number_chunks[z] for z in zbins]
    print("Nref = ", N_ref)
    for var in mean_pk_list_variation:
        N_var = [var.number_chunks[z] for z in zbins]
        ratio_ref = [var.number_chunks[z] / mean_pk_ref.number_chunks[z] for z in zbins]
        print("Nvar = ", N_var)
        print("Nvar / Nref = ", ratio_ref)
        print("mean Nvar / Nref = ", np.mean(ratio_ref))
