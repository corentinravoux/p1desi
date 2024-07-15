import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from p1desi import corrections, hcd, pk_io, utils


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


def plot_stat_uncertainties(file_pk, outname, zmax, **plot_args):
    pk = pk_io.Pk.read_from_picca(file_pk)

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

    A_lines = corrections.prepare_lines_correction(pk.zbin, lines_coeff_fit)
    A_hcd = corrections.prepare_hcd_correction(pk.zbin, hcd_coeff_fit)
    A_cont = corrections.prepare_cont_correction(pk.zbin, continuum_coeff_fit)
    A_reso = corrections.prepare_resolution_correction(pk.zbin, resolution_coeff_fit)

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

            syste_resocorrection[z] = 0.3 * np.abs(A_reso[z](pk.k[z]) - 1) * pk.p[z]
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


def plot_covariance(
    file_pk,
    zmax,
    kmax_AA,
    kmin_AA,
    out_name=None,
    out_points=None,
    use_boot=True,
    add_systematics=True,
    systematics_file=None,
    **plot_args,
):
    figsize = utils.return_key(plot_args, "figsize", (20, 20))
    wspace = utils.return_key(plot_args, "wspace", 0.3)
    subplot_x = utils.return_key(plot_args, "subplot_x", 4)
    subplot_y = utils.return_key(plot_args, "subplot_y", 4)
    vmin = utils.return_key(plot_args, "vmin", -1)
    vmax = utils.return_key(plot_args, "vmax", 1)
    cmap = utils.return_key(plot_args, "map_color", "seismic")

    pk = pk_io.Pk.read_from_picca(file_pk)

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
    fig.subplots_adjust(wspace=wspace)
    for j, z in enumerate(pk.zbin):
        if z < zmax:
            if pk.velunits:
                kmax = float(utils.kAAtokskm(kmax_AA, z=z))
                kmin = float(utils.kAAtokskm(kmin_AA, z=z))
            else:
                kmax = kmax_AA
                kmin = kmin_AA

            mask_cov = (pk.cov_k1[z] < kmax) & (pk.cov_k2[z] < kmax)
            mask_cov &= (pk.cov_k1[z] > kmin) & (pk.cov_k2[z] > kmin)

            mask = (pk.k[z] > kmin) & (pk.k[z] < kmax)
            nkbin = len(pk.k[z][mask])
            kmin_plot = np.min(pk.k[z][mask])
            kmax_plot = np.max(pk.k[z][mask])
            extent = [kmin_plot, kmax_plot, kmax_plot, kmin_plot]
            if use_boot:
                cov_mat = pk.boot_cov[z][mask_cov].reshape(nkbin, nkbin)
            else:
                cov_mat = pk.cov[z][mask_cov].reshape(nkbin, nkbin)
            if add_systematics:
                for i in range(len(list_systematics)):
                    cov_sys = np.outer(
                        list_systematics[i][z][mask], list_systematics[i][z][mask]
                    )
                    cov_mat = cov_mat + cov_sys

            mean_k1 = np.array([pk.k[z][mask] for i in range(len(pk.k[z][mask]))]).T
            mean_k2 = np.array([pk.k[z][mask] for i in range(len(pk.k[z][mask]))])

            v = np.sqrt(np.diag(cov_mat))
            outer_v = np.outer(v, v)
            corr_mat = cov_mat / outer_v

            ax = fig.add_subplot(subplot_y, subplot_x, j + 1)
            ax.set_title(f"Correlation matrix at z = {z}")
            im = ax.imshow(corr_mat, extent=extent,vmin=vmin,vmax=vmax,cmap=cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            z_arr.append(np.full((nkbin * nkbin), z))
            k1_arr.append(np.ravel(mean_k1))
            k2_arr.append(np.ravel(mean_k2))
            cov_mat_arr.append(np.ravel(cov_mat))
            corr_mat_arr.append(np.ravel(corr_mat))

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
