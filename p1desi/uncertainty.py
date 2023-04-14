import pickle
import matplotlib.pyplot as plt
from p1desi import utils, hcd
from matplotlib import cm
import numpy as np


def create_uncertainty_systematics(total_systematics, name):
    pickle.dump(total_systematics, open(name, "wb"))


def prepare_uncertainty_systematics(
    zbins,
    file_systematics,
):
    param_syst = pickle.load(open(file_systematics, "rb"))
    err_systematics = {}
    for iz, z in enumerate(zbins):
        err_systematics[z] = param_syst[iz]
    return err_systematics


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

    syste_tot = [[] for i in range(len(pk.zbin[pk.zbin < zmax]))]

    for iz, z in enumerate(pk.zbin):
        if z < zmax:
            syste_noise = 0.3 * (pk.p_noise_miss[z] / pk.resocor[z])
            syste_tot[iz].append(syste_noise**2)
            ax[0][1].plot(pk.k[z], syste_noise / pk.err[z], color=colors[iz])
            ax[0][0].plot(pk.k[z], syste_noise, color=colors[iz])
            ax[0][0].set_title(
                "Noise estimation", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_reso = 2 * pk.k[z] ** 2 * delta_l[iz] * delta_delta_l[iz] * pk.p[z]
            syste_tot[iz].append(syste_reso**2)
            ax[1][1].plot(pk.k[z], syste_reso / pk.err[z], color=colors[iz])
            ax[1][0].plot(pk.k[z], syste_reso, color=colors[iz])
            ax[1][0].set_title(
                "Resolution", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_resocorrection = (
                0.3 * np.abs(np.poly1d(resolution_coeff_fit)(pk.k[z]) - 1) * pk.p[z]
            )
            syste_tot[iz].append(syste_resocorrection**2)
            ax[2][1].plot(pk.k[z], syste_resocorrection / pk.err[z], color=colors[iz])
            ax[2][0].plot(pk.k[z], syste_resocorrection, color=colors[iz])
            ax[2][0].set_title(
                "Resolution Correction",
                x=title_shift,
                y=title_yshift,
                fontsize=title_size,
            )

            syste_sb = 0.3 * pk_sb.err[z]
            syste_tot[iz].append(syste_sb**2)
            ax[3][1].plot(pk.k[z], syste_sb / pk.err[z], color=colors[iz])
            ax[3][0].plot(pk.k[z], syste_sb, color=colors[iz])
            ax[3][0].set_title(
                "Side band", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_lines = (
                0.3 * np.abs(np.poly1d(lines_coeff_fit[iz])(pk.k[z]) - 1) * pk.p[z]
            )
            syste_tot[iz].append(syste_lines**2)
            ax[4][1].plot(
                pk.k[z],
                syste_lines / pk.err[z],
                label=r"$z = ${:1.1f}".format(z),
                color=colors[iz],
            )
            ax[4][0].plot(pk.k[z], syste_lines, color=colors[iz])
            ax[4][0].set_title(
                "Line masking", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_hcd = 0.3 * np.abs(hcd_coeff_fit[iz] - 1) * pk.p[z]
            syste_tot[iz].append(syste_hcd**2)
            ax[5][1].plot(pk.k[z], syste_hcd / pk.err[z], color=colors[iz])
            ax[5][0].plot(pk.k[z], syste_hcd, color=colors[iz])
            ax[5][0].set_title(
                "DLA masking", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_continuum = (
                0.3 * np.abs(np.poly1d(continuum_coeff_fit[iz])(pk.k[z]) - 1) * pk.p[z]
            )
            syste_tot[iz].append(syste_continuum**2)
            ax[6][1].plot(pk.k[z], syste_continuum / pk.err[z], color=colors[iz])
            ax[6][0].plot(pk.k[z], syste_continuum, color=colors[iz])
            ax[6][0].set_title(
                "Continuum fitting", x=title_shift, y=title_yshift, fontsize=title_size
            )

            A_dla_completeness = hcd.rogers(z, pk.k[z], *dla_completeness_coef[iz])
            syste_dla_completeness = 0.2 * np.abs(A_dla_completeness - 1) * pk.p[z]
            syste_tot[iz].append(syste_dla_completeness**2)
            ax[7][1].plot(pk.k[z], syste_dla_completeness / pk.err[z], color=colors[iz])
            ax[7][0].plot(pk.k[z], syste_dla_completeness, color=colors[iz])
            ax[7][0].set_title(
                "DLA completeness", x=title_shift, y=title_yshift, fontsize=title_size
            )

            A_dla_completeness = hcd.rogers(z, pk.k[z], *dla_completeness_coef[iz])
            syste_dla_completeness = 0.2 * np.abs(A_dla_completeness - 1) * pk.p[z]
            syste_tot[iz].append(syste_dla_completeness**2)
            ax[7][1].plot(pk.k[z], syste_dla_completeness / pk.err[z], color=colors[iz])
            ax[7][0].plot(pk.k[z], syste_dla_completeness, color=colors[iz])
            ax[7][0].set_title(
                "DLA completeness", x=title_shift, y=title_yshift, fontsize=title_size
            )

            syste_tot[iz] = np.sqrt(np.sum(syste_tot[iz], axis=0))
            ax[8][1].plot(pk.k[z], syste_tot[iz] / pk.err[z], color=colors[iz])
            ax[8][0].plot(pk.k[z], syste_tot[iz], color=colors[iz])
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
        pickle.dump(syste_tot, open(f"{outname}_kms.pickle", "wb"))
    else:
        plt.savefig(f"{outname}.pdf")
        pickle.dump(syste_tot, open(f"{outname}.pickle", "wb"))
