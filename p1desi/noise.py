from p1desi import utils, pk_io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.patches as mpatches


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
        mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk)
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


def plot_noise_comparison(
    pk,
    out_name=None,
    out_points=None,
    **kwargs,
):
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
    **kwargs,
):
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

    mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk)

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
        mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk)
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
):
    mean_pk_z = pk_io.MeanPkZ.init_from_pk(pk)
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
        linestype="None",
    )
    ax[0].errorbar(
        mean_pk_k.zbin,
        [mean_pk_k.p_diff[z] for z in mean_pk_k.zbin],
        [mean_pk_k.err_diff[z] for z in mean_pk_k.zbin],
        marker=marker,
        markerfacecolor="None",
        markeredgecolor=color,
        linestype="None",
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
