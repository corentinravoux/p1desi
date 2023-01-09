#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""


import numpy as np
import matplotlib.pyplot as plt
import fitsio
import seaborn as sns
import pandas
from matplotlib.ticker import FuncFormatter
from functools import partial
import scipy
import struct
import astropy.table as t

from p1desi import utils


def read_pk_means(pk_means_name, old_format=False):
    if old_format:
        pkmeans = t.Table.read(pk_means_name)
    else:
        pkmeans = t.Table.read(pk_means_name, hdu=1)
    return pkmeans


def prepare_plot_values(
    data,
    zbins,
    comparison=None,
    comparison_model=None,
    comparison_model_file=None,
    plot_P=False,
    z_binsize=0.2,
    velunits=False,
    substract_sb=None,
    substract_sb_comparison=True,
    beta_correction=None,
    beta_correction_sb=None,
):

    dict_plot = {}

    if plot_P:
        meanvar = "meanPk"
        errvar = "errorPk"
    else:
        meanvar = "meanDelta2"
        errvar = "errorDelta2"

    if comparison_model is not None:
        zmodel, kmodel, kpkmodel = load_model(comparison_model, comparison_model_file)
    minrescor = np.inf
    maxrescor = 0.0

    for iz, z in enumerate(zbins):
        dict_plot[z] = {}
        dat = data[iz]
        select = dat["N"] > 0
        if substract_sb is not None:
            dat_sb = substract_sb[iz]
            select_sb = dat_sb["N"] > 0
            p_sb = dat_sb[meanvar][select_sb]
            if beta_correction_sb is not None:
                p_sb = (
                    p_sb
                    - ((1 - beta_correction_sb) / beta_correction_sb)
                    * dat_sb["meanPk_noraw"][select_sb]
                )
        k_to_plot = np.array(dat["meank"][select])
        p_to_plot = dat[meanvar][select]
        if beta_correction is not None:
            p_to_plot = (
                p_to_plot
                - ((1 - beta_correction) / beta_correction)
                * dat["meanPk_noraw"][select]
            )
        if substract_sb is not None:
            p_to_plot = p_to_plot - p_sb
        err_to_plot = dat[errvar][select]

        if (comparison_model is not None) & (comparison is not None):
            raise ValueError(
                "Please choose between plotting a model or another P1D as a comparison"
            )

        if comparison_model is not None:
            izmodel = np.abs((zmodel - z)) < z_binsize / 2
            izmodel = izmodel.nonzero()[0][0]
            if velunits:
                convfactor = 1
            else:
                convfactor = 3e5 / (1215.67 * (1 + zmodel[izmodel, 0]))
            if plot_P:
                k_to_plot_comparison = kmodel[izmodel, :] * convfactor
                p_to_plot_comparison = (
                    (1 / convfactor) * kpkmodel[izmodel, :] / kmodel[izmodel, :] * np.pi
                )
            else:
                k_to_plot_comparison = kmodel[izmodel, :] * convfactor
                p_to_plot_comparison = kpkmodel[izmodel, :]
            err_to_plot_comparison = None

        if comparison is not None:
            k_to_plot_comparison = comparison["meank"][iz, :]
            p_to_plot_comparison = comparison[meanvar][iz, :]
            if substract_sb is not None:
                if substract_sb_comparison:
                    p_to_plot_comparison = p_to_plot_comparison - p_sb
            err_to_plot_comparison = comparison[errvar][iz, :]

        ## Comparison

        if (comparison_model is None) & (comparison is None):
            k_to_plot_comparison = None
            p_to_plot_comparison = None
            err_to_plot_comparison = None
            diff_k_to_plot = None
            diff_p_to_plot = None
            chi_p_to_plot = None
            diff_err_to_plot = None
        else:

            inter = scipy.interpolate.interp1d(
                k_to_plot_comparison, p_to_plot_comparison, fill_value="extrapolate"
            )
            p_comparison_interp = inter(k_to_plot)
            diff_k_to_plot = k_to_plot
            diff_p_to_plot = (p_to_plot - p_comparison_interp) / p_comparison_interp
            chi_p_to_plot = (p_to_plot - p_comparison_interp) / err_to_plot
            if err_to_plot_comparison is None:
                diff_err_to_plot = err_to_plot / p_comparison_interp
            else:
                inter_err = scipy.interpolate.interp1d(
                    k_to_plot_comparison,
                    err_to_plot_comparison,
                    fill_value="extrapolate",
                )
                err_comparison_interp = inter_err(k_to_plot)
                diff_err_to_plot = (p_to_plot / p_comparison_interp) * np.sqrt(
                    (err_to_plot / p_to_plot) ** 2
                    + (err_comparison_interp / p_comparison_interp) ** 2
                )

        if "rescor" in dat.colnames:
            try:
                if np.max(dict_plot[z]["k_to_plot"]) > 0:
                    minrescor = np.min(
                        [
                            minrescor,
                            np.min(
                                dict_plot[z]["k_to_plot"][
                                    (dat["rescor"][select] < 0.1)
                                    & (dat["rescor"][select] > 0)
                                ]
                            ),
                        ]
                    )
                    maxrescor = np.max(
                        [
                            maxrescor,
                            np.min(
                                dict_plot[z]["k_to_plot"][
                                    (dat["rescor"][select] < 0.1)
                                    & (dat["rescor"][select] > 0)
                                ]
                            ),
                        ]
                    )
            except:
                print("rescor information not computed, skipping")

        dict_plot["minrescor"] = minrescor
        dict_plot["maxrescor"] = maxrescor

        dict_plot[z]["number_chunks"] = dat["N_chunks"]
        dict_plot[z]["k_to_plot"] = k_to_plot
        dict_plot[z]["p_to_plot"] = p_to_plot
        dict_plot[z]["err_to_plot"] = err_to_plot
        dict_plot[z]["k_to_plot_comparison"] = k_to_plot_comparison
        dict_plot[z]["p_to_plot_comparison"] = p_to_plot_comparison
        dict_plot[z]["err_to_plot_comparison"] = err_to_plot_comparison
        dict_plot[z]["diff_k_to_plot"] = diff_k_to_plot
        dict_plot[z]["diff_p_to_plot"] = diff_p_to_plot
        dict_plot[z]["chi_p_to_plot"] = chi_p_to_plot
        dict_plot[z]["diff_err_to_plot"] = diff_err_to_plot

    return dict_plot


def return_mean_z_dict(zbins, data):
    mean_dict = {
        "meanPk_diff": [],
        "meanPk_noise": [],
        "meanPk_raw": [],
        "error_diffovernoise": [],
        "error_meanPk_diffoverraw": [],
        "error_meanPk_noiseoverraw": [],
        "error_meanPk_raw": [],
        "k_array": [],
        "error_meanPk_noise": [],
        "errorPk": [],
        "meanPk": [],
    }
    for z, d in zip(zbins, data):
        mean_dict["meanPk"].append(d["meanPk"])
        mean_dict["errorPk"].append(d["errorPk"])
        mean_dict["meanPk_noise"].append(d["meanPk_noise"])
        mean_dict["meanPk_raw"].append(d["meanPk_raw"])
        yerr = (d["errorPk_noise"] / d["meanPk_raw"]) * np.sqrt(
            (d["errorPk_raw"] / d["meanPk_raw"]) ** 2
            + (d["errorPk_noise"] / d["meanPk_noise"]) ** 2
        )
        mean_dict["error_meanPk_noiseoverraw"].append(yerr)
        mean_dict["k_array"] = d["meank"]
        mean_dict["error_meanPk_raw"].append(d["errorPk_raw"])
        mean_dict["error_meanPk_noise"].append(d["errorPk_noise"])

        mean_dict["meanPk_diff"].append(d["meanPk_diff"])
        mean_dict["error_diffovernoise"].append(yerr)
        yerr = (d["errorPk_diff"] / d["meanPk_raw"]) * np.sqrt(
            (d["errorPk_raw"] / d["meanPk_raw"]) ** 2
            + (d["errorPk_diff"] / d["meanPk_diff"]) ** 2
        )
        mean_dict["error_meanPk_diffoverraw"].append(yerr)

    mean_dict["meanPk_noise"] = np.mean(mean_dict["meanPk_noise"], axis=0)
    mean_dict["error_meanPk_noiseoverraw"] = np.mean(
        mean_dict["error_meanPk_noiseoverraw"], axis=0
    ) / np.sqrt(len(mean_dict["error_meanPk_noiseoverraw"]))
    mean_dict["meanPk_raw"] = np.mean(mean_dict["meanPk_raw"], axis=0)
    mean_dict["error_meanPk_raw"] = np.mean(
        mean_dict["error_meanPk_raw"], axis=0
    ) / np.sqrt(len(mean_dict["error_meanPk_raw"]))
    mean_dict["meanPk"] = np.mean(mean_dict["meanPk"], axis=0)
    mean_dict["errorPk"] = np.mean(mean_dict["errorPk"], axis=0) / np.sqrt(
        len(mean_dict["errorPk"])
    )

    mean_dict["error_meanPk_noise"] = np.mean(
        mean_dict["error_meanPk_noise"], axis=0
    ) / np.sqrt(len(mean_dict["error_meanPk_noise"]))
    mean_dict["meanPk_diff"] = np.mean(mean_dict["meanPk_diff"], axis=0)
    mean_dict["error_meanPk_diffoverraw"] = np.mean(
        mean_dict["error_meanPk_diffoverraw"], axis=0
    ) / np.sqrt(len(mean_dict["error_meanPk_diffoverraw"]))

    mean_dict["error_diffovernoise"] = np.mean(
        mean_dict["error_diffovernoise"], axis=0
    ) / np.sqrt(len(mean_dict["error_diffovernoise"]))

    return mean_dict


def load_model(model, model_file):

    if model == "eBOSSmodel_stack":
        eBOSSmodel_lowz = read_in_model(model_file[0])
        eBOSSmodel_highz = read_in_model(model_file[1])
        eBOSSmodel_stack = [
            np.vstack([m, m2]) for m, m2 in zip(eBOSSmodel_lowz, eBOSSmodel_highz)
        ]
        return eBOSSmodel_stack
    elif model == "Naimmodel_stack":

        def naim_function4(
            k,
            z,
            k0=0.009,
            k1=0.053,
            z0=3,
            A=0.066,
            B=3.59,
            n=-2.685,
            alpha=-0.22,
            beta=-0.16,
        ):
            knorm0 = k / k0
            knorm1 = k / k1
            exp1 = 3 + n + alpha * np.log(knorm0)
            exp2 = B + beta * np.log(knorm0)
            nom = knorm0**exp1
            denom = 1 + knorm1**2
            zfac = (1 + z) / (1 + z0)
            return A * nom / denom * zfac**exp2

        Naimmodel = {}
        z_array = np.arange(2.2, 4.7, 0.2)
        k_array = np.arange(0.001, 0.1, 0.0001)
        Naimmodel["kpk"] = naim_function4(
            k_array[np.newaxis, :],
            z_array[:, np.newaxis],
            A=0.084,
            B=3.64,
            alpha=-0.155,
            beta=0.32,
            k1=0.048,
            n=-2.655,
        )
        kk, zz = np.meshgrid(k_array, z_array)
        Naimmodel["k"] = kk
        Naimmodel["z"] = zz

        Naimmodel_stack = (
            np.array(Naimmodel["z"]),
            np.array(Naimmodel["k"]),
            np.array(Naimmodel["kpk"]),
        )
        return Naimmodel_stack

    elif model == "Naimmodel_truth_mocks":

        def readTrueP1D(fname):
            file = open(fname, "rb")
            nk, nz = struct.unpack("ii", file.read(struct.calcsize("ii")))

            fmt = "d" * nz
            data = file.read(struct.calcsize(fmt))
            z = np.array(struct.unpack(fmt, data), dtype=np.double)

            fmt = "d" * nk
            data = file.read(struct.calcsize(fmt))
            k = np.array(struct.unpack(fmt, data), dtype=np.double)

            fmt = "d" * nk * nz
            data = file.read(struct.calcsize(fmt))
            p1d = np.array(struct.unpack(fmt, data), dtype=np.double).reshape((nz, nk))

            return z, k, p1d

        z, k, p = readTrueP1D(model_file)
        Naimmodel = {}
        Naimmodel["z"] = np.array(
            [[z[i] for j in range(len(k))] for i in range(len(z))]
        )
        Naimmodel["k"] = np.array([k for i in range(len(z))])
        Naimmodel["kpk"] = p * k / np.pi
        Naimmodel_mock = (
            np.array(Naimmodel["z"]),
            np.array(Naimmodel["k"]),
            np.array(Naimmodel["kpk"]),
        )
        return Naimmodel_mock
    else:
        raise ValueError("Incorrect model")


def read_in_model(filename):
    tab = fitsio.FITS(filename)[1]
    z = tab["z"][:].reshape(-1, 1000)
    k = tab["k"][:].reshape(-1, 1000)
    kpk = tab["kpk"][:].reshape(-1, 1000)
    return z, k, kpk


def plot_data(
    data,
    zbins,
    outname,
    plot_P=False,
    comparison=None,
    comparison_model=None,
    comparison_model_file=None,
    plot_diff=False,
    substract_sb=None,
    beta_correction=None,
    beta_correction_sb=None,
    **kwargs,
):

    velunits = data.meta["VELUNITS"]

    res_label = utils.return_key(kwargs, "res_label", "")
    res_label2 = utils.return_key(kwargs, "res_label2", "")
    diff_range = utils.return_key(kwargs, "diff_range", 0.4)
    no_errors_diff = utils.return_key(kwargs, "no_errors_diff", False)
    marker_size = utils.return_key(kwargs, "marker_size", 6)
    marker_style = utils.return_key(kwargs, "marker_style", "o")
    fonttext = utils.return_key(kwargs, "fonttext", None)
    fontlab = utils.return_key(kwargs, "fontlab", None)
    fontlegend = utils.return_key(kwargs, "fontl", None)
    z_binsize = utils.return_key(kwargs, "z_binsize", 0.2)
    colors = utils.return_key(kwargs, "colors", sns.color_palette("deep", len(zbins)))
    kmin = utils.return_key(kwargs, "kmin", 4e-2)
    kmax = utils.return_key(kwargs, "kmax", 2.5)
    ymin = utils.return_key(kwargs, "ymin", None)
    ymax = utils.return_key(kwargs, "ymax", None)
    grid = utils.return_key(kwargs, "grid", True)
    figsize = utils.return_key(kwargs, "figsize", (8, 8))
    substract_sb_comparison = utils.return_key(kwargs, "substract_sb_comparison", True)

    if comparison is not None:
        comparison_data = read_pk_means(comparison)
    else:
        comparison_data = None

    comparison_plot_style = utils.return_key(kwargs, "comparison_plot_style", None)

    dict_plot = prepare_plot_values(
        data,
        zbins,
        comparison=comparison_data,
        comparison_model=comparison_model,
        comparison_model_file=comparison_model_file,
        plot_P=plot_P,
        z_binsize=z_binsize,
        velunits=velunits,
        substract_sb=substract_sb,
        substract_sb_comparison=substract_sb_comparison,
        beta_correction=beta_correction,
        beta_correction_sb=beta_correction_sb,
    )
    if dict_plot[zbins[0]]["k_to_plot_comparison"] is not None:
        fig, (ax, ax2) = plt.subplots(
            2, figsize=figsize, gridspec_kw=dict(height_ratios=[3, 1]), sharex=True
        )
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    if dict_plot[zbins[0]]["k_to_plot_comparison"] is not None:
        if not velunits:
            par1, par2, par3 = utils.place_k_speed_unit_axis(fig, ax, ax2, fonttext)

    for iz, z in enumerate(zbins):
        ax.errorbar(
            dict_plot[z]["k_to_plot"],
            dict_plot[z]["p_to_plot"],
            yerr=dict_plot[z]["err_to_plot"],
            fmt=marker_style,
            color=colors[iz],
            markersize=marker_size,
            label=r" z = {:1.1f}, {} chunks".format(z, dict_plot[z]["number_chunks"]),
        )

        if dict_plot[z]["k_to_plot_comparison"] is not None:
            if (comparison_plot_style == "fill") & (
                dict_plot[z]["err_to_plot_comparison"] is not None
            ):
                ax.fill_between(
                    dict_plot[z]["k_to_plot_comparison"],
                    dict_plot[z]["p_to_plot_comparison"]
                    - dict_plot[z]["err_to_plot_comparison"],
                    dict_plot[z]["p_to_plot_comparison"]
                    + dict_plot[z]["err_to_plot_comparison"],
                    alpha=0.5,
                    color=colors[iz],
                )
            else:
                if dict_plot[z]["err_to_plot_comparison"] is not None:
                    ax.errorbar(
                        dict_plot[z]["k_to_plot_comparison"],
                        dict_plot[z]["p_to_plot_comparison"],
                        dict_plot[z]["err_to_plot_comparison"],
                        color=colors[iz],
                        ls=":",
                    )
                else:
                    ax.plot(
                        dict_plot[z]["k_to_plot_comparison"],
                        dict_plot[z]["p_to_plot_comparison"],
                        color=colors[iz],
                        ls=":",
                    )
            if no_errors_diff:
                ax2.plot(
                    dict_plot[z]["diff_k_to_plot"],
                    dict_plot[z]["diff_p_to_plot"],
                    color=colors[iz],
                    label="",
                    marker=".",
                    ls="",
                    zorder=-iz,
                )
            else:
                ax2.errorbar(
                    dict_plot[z]["diff_k_to_plot"],
                    dict_plot[z]["diff_p_to_plot"],
                    dict_plot[z]["diff_err_to_plot"],
                    color=colors[iz],
                    label="",
                    ls="",
                    marker=".",
                    zorder=-iz,
                )

    if (dict_plot["minrescor"] != np.inf) | (dict_plot["minrescor"] != 0.0):
        ax.fill_betweenx(
            [-1000, 1000],
            [dict_plot["minrescor"], dict_plot["minrescor"]],
            [dict_plot["maxrescor"], dict_plot["maxrescor"]],
            color="0.7",
            zorder=-30,
        )
        if dict_plot[z]["k_to_plot_comparison"] is not None:
            ax2.fill_betweenx(
                [-1000, 1000],
                [dict_plot["minrescor"], dict_plot["minrescor"]],
                [dict_plot["maxrescor"], dict_plot["maxrescor"]],
                color="0.7",
                zorder=-30,
                label="",
            )

    if dict_plot[z]["k_to_plot_comparison"] is not None:
        if velunits:
            ax2.set_xlabel(r" k [s/km]", fontsize=fonttext)
        else:
            ax2.set_xlabel(r" k [1/$\AA$]", fontsize=fonttext)

    else:
        if velunits:
            ax.set_xlabel(r" k [s/km]", fontsize=fonttext)
        else:
            ax.set_xlabel(r" k [1/$\AA$]", fontsize=fonttext)

    if plot_P:
        ax.set_ylabel(r"$P_{1d}$ ", fontsize=fonttext, labelpad=-1)
        if dict_plot[z]["k_to_plot_comparison"] is not None:
            ax2.set_ylabel(r"$(P_{1d,data}-P_{1d,comp})/P_{1d,comp}$")
    else:
        ax.set_ylabel(r"$\Delta^2_{1d}$ ", fontsize=fonttext, labelpad=-1)
        if dict_plot[z]["k_to_plot_comparison"] is not None:
            ax2.set_ylabel(
                r"$(\Delta^2_{1d,data}-\Delta^2_{1d,comp})/\Delta^2_{1d,comp}$"
            )

    ax.set_yscale("log")
    if grid:
        ax.grid()
    ax.xaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(direction="in")
    ax.yaxis.set_ticks_position("both")
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_tick_params(labelsize=fontlab)
    ax.yaxis.set_tick_params(labelsize=fontlab)
    ax.set_xlim(kmin, kmax)
    if dict_plot[z]["k_to_plot_comparison"] is not None:
        if grid:
            ax2.grid()
        ax2.xaxis.set_ticks_position("both")
        ax2.xaxis.set_tick_params(direction="in")
        ax2.yaxis.set_ticks_position("both")
        ax2.yaxis.set_tick_params(direction="in")
        ax2.xaxis.set_tick_params(labelsize=fontlab)
        ax2.yaxis.set_tick_params(labelsize=fontlab)
        ax2.set_xlim(kmin, kmax)

    if ymin is None:
        if not plot_P:
            ax.set_ylim(4e-3, 2)
        else:
            if not velunits:
                ax.set_ylim(0.01, 0.5)
            else:
                ax.set_ylim(1, 300)
    else:
        ax.set_ylim(ymin, ymax)

    if dict_plot[z]["k_to_plot_comparison"] is not None:
        ax2.set_ylim(-diff_range / 2, diff_range / 2)
    handles, labels = ax.get_legend_handles_labels()

    legend1 = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.03, -0.5, 0.4, 1.0),
        borderaxespad=0.0,
        fontsize=fontlegend,
    )

    ax.errorbar(
        [0],
        [0],
        yerr=[0],
        fmt=marker_style,
        color="k",
        markersize=marker_size,
        label="{}".format(res_label),
    )
    if comparison_plot_style == "fill":
        ax.fill_between([0], [0], [0], label=res_label2, color="k")
    else:
        ax.plot([0], [0], label=res_label2, color="k", ls=":")

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(
        *[(h, l) for (h, l) in zip(handles, labels) if not "z =" in l]
    )
    ax.legend(
        handles,
        labels,
        loc=2,
        bbox_to_anchor=(1.03, 0.9),
        borderaxespad=0.0,
        fontsize=fontlegend,
    )

    if dict_plot[z]["k_to_plot_comparison"] is not None:
        if not velunits:
            par1.set_xlim(*ax2.get_xlim())
            par2.set_xlim(*ax2.get_xlim())
            par3.set_xlim(*ax2.get_xlim())

    ax.add_artist(legend1)
    fig.subplots_adjust(
        top=0.75, bottom=0.1, left=0.1, right=0.65, hspace=0.2, wspace=0.2
    )
    fig.savefig(outname + f"{'' if not plot_P else '_powernotDelta'}_kmax_{kmax}.pdf")

    if plot_diff:
        plot_diff_figure(outname, zbins, dict_plot, kmax, colors, res_label, res_label2)


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


def plot_lines_study(multiple_data, zbins, out_name, k_units, **kwargs):
    for i in range(len(multiple_data)):
        mean_dict = return_mean_z_dict(zbins, multiple_data[i])
        mean_dict["k_array"], mean_dict["meanPk"]

    return ()


# Uncertainties plots


def plot_uncertainties(data, zbins, out_name, k_units, **kwargs):
    return ()
