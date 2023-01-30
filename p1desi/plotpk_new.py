#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from p1desi import utils, uncertainty


# CR - once all plotting functions in notebooks are adapted, add them here


def plot(
    pk,
    zmax,
    outname=None,
    outpoints=None,
    plot_P=False,
    systematics_file=None,
    **plot_args,
):

    zbins = pk.zbin[pk.zbin < zmax]

    marker_size = utils.return_key(plot_args, "marker_size", 7)
    marker_style = utils.return_key(plot_args, "marker_style", ".")
    fontsize_x = utils.return_key(plot_args, "fontsize_x", 16)
    fontsize_y = utils.return_key(plot_args, "fontsize_y", 19)
    labelsize = utils.return_key(plot_args, "labelsize", 14)
    fontlegend = utils.return_key(plot_args, "fontl", 14)
    color = utils.return_key(plot_args, "color", [f"C{i}" for i in range(len(zbins))])
    kmin_AA = utils.return_key(plot_args, "kmin_AA", 0.145)
    kmax_AA = utils.return_key(plot_args, "kmax_AA", 2.5)
    ymin = utils.return_key(plot_args, "ymin", 0.01)
    ymax = utils.return_key(plot_args, "ymax", 0.2)
    figsize = utils.return_key(plot_args, "figsize", (11, 8.5))
    place_velaxis = utils.return_key(plot_args, "place_velaxis", True)

    fig, ax = plt.subplots(1, figsize=figsize)

    if systematics_file is not None:
        systematics = uncertainty.prepare_uncertainty_systematics(
            zbins, systematics_file
        )

    zarr, karr, pkarr, errarr = [], [], [], []
    for i, z in enumerate(zbins):
        if systematics_file is not None:
            error_bar = np.sqrt(pk.err[z] ** 2 + systematics[z] ** 2)
        else:
            error_bar = pk.err[z]

        if pk.velunits:
            kmax = float(utils.kAAtokskm(kmax_AA, z=z))
            kmin = float(utils.kAAtokskm(kmin_AA, z=z))
        else:
            kmax = kmax_AA
            kmin = kmin_AA

        mask_k = (pk.k[z] < kmax) & (pk.k[z] > kmin)
        k = pk.k[z][mask_k]

        if plot_P:
            p = pk.p[z][mask_k]
            err = error_bar[mask_k]

        else:
            p = pk.norm_p[z][mask_k]
            err = (pk.k[z] * error_bar / np.pi)[mask_k]

        zarr.append(np.array([z for j in range(len(k))]))
        karr.append(k)
        pkarr.append(p)
        errarr.append(err)
        ax.errorbar(
            k,
            p,
            yerr=err,
            fmt=marker_style,
            color=color[i],
            markersize=marker_size,
            label=r"$z = ${:1.1f}  ({} chunks)".format(z, pk.number_chunks[z]),
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
        zarr = np.around(np.concatenate(zarr, axis=0), 3)
        karr = np.concatenate(karr, axis=0)
        pkarr = np.concatenate(pkarr, axis=0)
        errarr = np.concatenate(errarr, axis=0)
        text_file = np.vstack([zarr, karr, pkarr, errarr])
        np.savetxt(outpoints, np.transpose(text_file))

    return (fig, ax)


def plot_comparison(
    pk,
    pk2,
    zmax,
    outname=None,
    outpoints=None,
    plot_P=False,
    systematics_file=None,
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
    color = utils.return_key(plot_args, "color", [f"C{i}" for i in range(len(zbins))])
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

    fig, ax = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[3, 1]), sharex=True
    )

    if systematics_file is not None:
        systematics = uncertainty.prepare_uncertainty_systematics(
            zbins, systematics_file
        )

    zarr, karr, pkarr, errarr = [], [], [], []
    zarr2, karr2, pkarr2, errarr2 = [], [], [], []
    for i, z in enumerate(zbins):
        if systematics_file is not None:
            error_bar = np.sqrt(pk.err[z] ** 2 + systematics[z] ** 2)
        else:
            error_bar = pk.err[z]

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
        else:
            mask_k2 = np.full(pk2.k[z].shape, True)
        k2 = pk2.k[z][mask_k2]

        if plot_P:
            p = pk.p[z][mask_k]
            err = error_bar[mask_k]
            p2 = pk2.p[z][mask_k2]
            err2 = pk2.err[z][mask_k2]

        else:
            p = pk.norm_p[z][mask_k]
            err = (pk.k[z] * error_bar / np.pi)[mask_k]
            p2 = pk2.norm_p[z][mask_k2]
            err2 = pk2.norm_err[z][mask_k2]

        zarr.append(np.array([z for j in range(len(k))]))
        karr.append(k)
        pkarr.append(p)
        errarr.append(err)

        zarr2.append(np.array([z for j in range(len(k2))]))
        karr2.append(k2)
        pkarr2.append(p2)
        errarr2.append(err2)
        ax[0].errorbar(
            k,
            p,
            yerr=err,
            marker=marker_style,
            linestyle="None",
            color=color[i],
            markersize=marker_size,
        )
        if marker_comp is not None:
            ax[0].errorbar(
                k2,
                p2,
                err2,
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
                p2 - err2,
                p2 + err2,
                color=color[i],
                alpha=alpha_comp,
            )

            ax[0].errorbar(
                k2,
                p2,
                err2,
                marker="None",
                color=color[i],
                markersize=marker_size,
                alpha=alpha_comp,
            )

        p2_interp = interp1d(
            k2,
            p2,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )(k)
        err_p2_interp = interp1d(
            k2,
            err2,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )(k)

        ratio = p / p2_interp
        err_ratio = (p / p2_interp) * np.sqrt(
            (err / p) ** 2 + (err_p2_interp / p2_interp) ** 2
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
        zarr = np.around(np.concatenate(zarr, axis=0), 3)
        karr = np.concatenate(karr, axis=0)
        pkarr = np.concatenate(pkarr, axis=0)
        errarr = np.concatenate(errarr, axis=0)
        text_file = np.vstack([zarr, karr, pkarr, errarr])
        np.savetxt(outpoints + ".txt", np.transpose(text_file))

        zarr2 = np.around(np.concatenate(zarr2, axis=0), 3)
        karr2 = np.concatenate(karr2, axis=0)
        pkarr2 = np.concatenate(pkarr2, axis=0)
        errarr2 = np.concatenate(errarr2, axis=0)
        text_file2 = np.vstack([zarr2, karr2, pkarr2, errarr2])
        np.savetxt(outpoints + "_2.txt", np.transpose(text_file2))

    return (fig, ax)


"""
def plot_diff_figure(outname, zbins, dict_plot, kmax, colors, reslabel, reslabel2):
    diff_data_model = []
    chi_data_model = []
    mask_k = dict_plot[zbins[0]]["diff_k_to_plot"] < kmax
    for iz, z in enumerate(zbins):
        diff_data_model.append(dict_plot[z]["diff_p_to_plot"][mask_k])
        chi_data_model.append(dict_plot[z]["chi_p_to_plot"][mask_k])
    plt.figure()
    sns.violinplot(
        data=pandas.DataFrame(np.array(diff_data_model).T, None, zbins),
        inner=None,
        orient="v",
        palette=colors,
        scale="width",
    )
    for i, d in enumerate(diff_data_model):
        plt.errorbar(i, np.mean(d), scipy.stats.sem(d, ddof=0), color="0.3", marker=".")
    plt.xlabel("z")
    plt.ylabel("$(P-P_{model})/P$")
    plt.tight_layout()
    plt.savefig(
        outname
        + f"_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}_diff.pdf"
    )
    plt.figure()
    sns.violinplot(
        data=pandas.DataFrame(np.array(chi_data_model).T, None, zbins),
        inner=None,
        orient="v",
        palette=colors,
        scale="width",
    )
    for i, d in enumerate(chi_data_model):
        plt.errorbar(i, np.mean(d), scipy.stats.sem(d, ddof=0), color="0.3", marker=".")
    plt.xlabel("z")
    plt.ylabel("$(P-P_{model})/\sigma_P}$")
    plt.tight_layout()
    plt.savefig(
        outname
        + f"_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}_chi.pdf"
    )



# Line plots


def plot_lines_study(multiple_data, zbins, out_name, k_units, **plot_args):
    for i in range(len(multiple_data)):
        mean_dict = return_mean_z_dict(zbins, multiple_data[i])
        mean_dict["k_array"], mean_dict["meanPk"]

    return ()


 """
