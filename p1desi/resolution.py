import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pickle
from p1desi import utils


def plot_mean_resolution(
    pk,
    zmax,
    outfig,
    outpoints,
    kmax_line=None,
    **kwargs,
):

    style = utils.return_key(kwargs, "style", None)
    if style is not None:
        plt.style.use(style)
    figsize = utils.return_key(kwargs, "figsize", (7, 5))
    size_label = utils.return_key(kwargs, "size", 14)
    size_legend = utils.return_key(kwargs, "size_legend", 14)
    size_font_x = utils.return_key(kwargs, "size_font_x", 14)
    size_font_y = utils.return_key(kwargs, "size_font_y", 16)

    plt.figure(figsize=figsize)

    mean_k, mean_reso, mean_pixelization = [], [], []

    for i, z in enumerate(pk.zbin):
        if z < zmax:
            pixelization = (np.sin(pk.k[z] * 0.4) / (0.4 * pk.k[z])) ** 2
            mean_k.append(pk.k[z])
            mean_reso.append(pk.resocor[z])
            mean_pixelization.append(pixelization)

    mean_k = np.mean(mean_k, axis=0)
    mean_reso = np.mean(mean_reso, axis=0)
    mean_pixelization = np.mean(mean_pixelization, axis=0)

    plt.plot(mean_k, mean_reso, marker=".", linestyle="None")
    plt.plot(mean_k, mean_pixelization)
    plt.legend(["Resolution & Pixelization", "Pixelization only"], fontsize=size_legend)

    for z in pk.zbin:
        if z < zmax:
            plt.plot(
                pk.k[z],
                pk.resocor[z],
                marker="None",
                linestyle="--",
                color="k",
                alpha=0.2,
            )

    if kmax_line is not None:
        plt.axvline(kmax_line, color="k")
    plt.fill_between(np.linspace(0, np.max(mean_k), 10), 0.0, 0.2, alpha=0.2, color="k")
    if pk.velunits:
        plt.xlabel(r"$k~[\AA^{-1}]$", fontsize=size_font_x)
    else:
        plt.xlabel(
            r"$k~[\mathrm{s}$" + r"$\cdot$" + "$\mathrm{km}^{-1}]$",
            fontsize=size_font_x,
        )

    plt.ylabel("Average correction", fontsize=size_font_y)

    plt.ylim(bottom=0)
    plt.gca().margins(x=0)
    plt.gca().tick_params("x", labelsize=size_label)
    plt.gca().tick_params("y", labelsize=size_label)

    plt.tight_layout()
    if outfig is not None:
        plt.savefig(outfig)
    if outpoints is not None:
        text_file = np.vstack([mean_k, mean_reso, mean_pixelization])
        np.savetxt(outpoints + ".txt", np.transpose(text_file))

    mask_k_90 = mean_reso < 0.1
    mask_k_95 = mean_reso < 0.05

    pk_cut_90 = np.min(mean_k[mask_k_90])
    pk_cut_95 = np.min(mean_k[mask_k_95])

    return pk_cut_90, pk_cut_95


def model_resolution(k, dl):
    resolution = np.exp(-0.5 * (k * dl) ** 2) * (np.sin(k * 0.4) / (k * 0.4))
    return resolution


def fit_model_resolution(nb_bins, x, y, dy):
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy = x[mask], y[mask], dy[mask]
    popt, _ = curve_fit(
        model_resolution,
        xdata=x,
        ydata=y,
        sigma=dy,
        p0=[0.8],
        bounds=([-np.inf], [np.inf]),
    )
    x_fit = np.linspace(x.min(), x.max(), nb_bins)

    def model_fit(x):
        return model_resolution(x, *popt)

    return (x_fit, model_fit, popt)


def fit_resolution_redshift(pk, zmax, outfile):
    delta_l = []

    for i, z in enumerate(pk.zbin):
        if z < zmax:
            x_fit, m_fit, popt = fit_model_resolution(
                500, pk.k[z], pk.resocor[z], pk.err_resocor[z]
            )
            delta_l.append(popt[0])

    pickle.dump(delta_l, open(outfile, "wb"))
