import pickle
import matplotlib.pyplot as plt
from p1desi import utils, hcd
from matplotlib import cm
import numpy as np


def create_uncertainty_systematics(
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
    pickle.dump(
        (
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
    zbins,
    file_systematics,
):
    (
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
    return syste_tot, list_systematics, list_systematics_name


def plot_stat_uncertainties(pk, outname, zmax, **plot_args):

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
    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        colors = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        colors = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    for i, z in enumerate(pk.zbin):
        if z < zmax:

            if pk.velunits:
                kmax = float(utils.kAAtokskm(kmax_AA, z=z))
                kmin = float(utils.kAAtokskm(kmin_AA, z=z))
            else:
                kmax = kmax_AA
                kmin = kmin_AA

            mask = (pk.k[z] > kmin) & (pk.k[z] < kmax)
            ax[0].semilogy(
                pk.k[z][mask],
                pk.err[z][mask],
                label=r"$z = ${:1.1f}".format(z),
                color=colors[i],
            )
            ax[1].semilogy(
                pk.k[z][mask],
                pk.err[z][mask] / pk.p[z][mask],
                label=r"$z = ${:1.1f}".format(z),
                color=colors[i],
            )
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
    fig.tight_layout()
    fig.savefig(f"{outname}.pdf")
    fig.savefig(f"{outname}.png")


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

    color_map = utils.return_key(plot_args, "color_map", "default")
    if color_map == "default":
        colors = [f"C{i}" for i, z in enumerate(pk.zbin) if z < zmax]
    elif color_map == "rainbow":
        colors = cm.rainbow(np.linspace(0, 1, len(pk.zbin[pk.zbin < zmax])))

    fig, ax = plt.subplots(n_subplots, 2, figsize=figsize, sharex=True)

    (
        syste_noise,
        syste_reso,
        syste_resocorrection,
        syste_sb,
        syste_lines,
        syste_hcd,
        syste_continuum,
        syste_dla_completeness,
        syste_tot,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {})

    for iz, z in enumerate(pk.zbin):
        if z < zmax:
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

            syste_resocorrection[z] = (
                0.3 * np.abs(np.poly1d(resolution_coeff_fit)(pk.k[z]) - 1) * pk.p[z]
            )
            syste_tot[z].append(syste_resocorrection[z] ** 2)
            ax[2][1].plot(
                pk.k[z], syste_resocorrection[z] / pk.err[z], color=colors[iz]
            )
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

            syste_lines[z] = (
                0.3 * np.abs(np.poly1d(lines_coeff_fit[iz])(pk.k[z]) - 1) * pk.p[z]
            )
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

            syste_hcd[z] = 0.3 * np.abs(hcd_coeff_fit[iz] - 1) * pk.p[z]
            syste_tot[z].append(syste_hcd[z] ** 2)
            ax[5][1].plot(pk.k[z], syste_hcd[z] / pk.err[z], color=colors[iz])
            ax[5][0].plot(pk.k[z], syste_hcd[z], color=colors[iz])
            ax[5][0].set_title(
                "DLA masking", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_continuum[z] = (
                0.3 * np.abs(np.poly1d(continuum_coeff_fit[iz])(pk.k[z]) - 1) * pk.p[z]
            )
            syste_tot[z].append(syste_continuum[z] ** 2)
            ax[6][1].plot(pk.k[z], syste_continuum[z] / pk.err[z], color=colors[iz])
            ax[6][0].plot(pk.k[z], syste_continuum[z], color=colors[iz])
            ax[6][0].set_title(
                "Continuum fitting", x=title_shift, y=title_yshift, fontsize=title_size
            )

            A_dla_completeness = hcd.rogers(z, pk.k[z], *dla_completeness_coef[iz])
            syste_dla_completeness[z] = 0.2 * np.abs(A_dla_completeness - 1) * pk.p[z]
            syste_tot[z].append(syste_dla_completeness[z] ** 2)
            ax[7][1].plot(
                pk.k[z], syste_dla_completeness[z] / pk.err[z], color=colors[iz]
            )
            ax[7][0].plot(pk.k[z], syste_dla_completeness[z], color=colors[iz])
            ax[7][0].set_title(
                "DLA completeness", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_tot[z] = np.sqrt(np.sum(syste_tot[z], axis=0))
            ax[8][1].plot(pk.k[z], syste_tot[z] / pk.err[z], color=colors[iz])
            ax[8][0].plot(pk.k[z], syste_tot[z], color=colors[iz])
            ax[8][0].set_title(
                "Total", x=title_shift, y=title_yshift, fontsize=title_size
            )

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
