import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit


def compute_chi2(
    mean_pk_list,
    mean_pk_ref,
    kmin=0.0,
    kmax=2.0,
    zmin=2.1,
    zmax=4.3,
    use_boot=True,
    extra_diag_errors=None,
    extra_cov_factor=None,
    use_only_diag=False,
    return_redshift_chi2=False,
):

    zbin = mean_pk_ref.zbin[(mean_pk_ref.zbin > zmin) & (mean_pk_ref.zbin < zmax)]

    if return_redshift_chi2:
        chi2_list, ndof_list = [[] for _ in enumerate(zbin)], [
            [] for _ in enumerate(zbin)
        ]
    else:
        chi2_list, ndof_list = [], []

    for mean_pk in mean_pk_list:
        chi2 = 0.0
        ndof = 0
        for i, z in enumerate(zbin):
            mask = (mean_pk.k[z] > kmin) & (mean_pk.k[z] < kmax)
            k = mean_pk.k[z][mask]

            if use_only_diag:
                diag = mean_pk.err[z][mask] ** 2
                cov = np.diag(diag)
                vector = mean_pk.p[z][mask] - mean_pk_ref.p[z][mask]
            else:
                if use_boot:
                    cov = mean_pk.boot_cov[z]
                else:
                    cov = mean_pk.cov[z]
                k1 = mean_pk.cov_k1[z]
                k2 = mean_pk.cov_k2[z]

                k1_matrix = k1.reshape(int(np.sqrt(k1.size)), int(np.sqrt(k1.size)))

                mask_pk = (k1_matrix[:, 0] > kmin) & (k1_matrix[:, 0] < kmax)

                mask_cov = (k1 < kmax) & (k2 < kmax)
                mask_cov &= (k1 > kmin) & (k2 > kmin)
                cov = cov[mask_cov]

                cov = cov.reshape(int(np.sqrt(cov.size)), int(np.sqrt(cov.size)))

                vector = mean_pk.p[z][mask_pk] - mean_pk_ref.p[z][mask_pk]

            if extra_cov_factor is not None:
                if type(extra_cov_factor) == dict:
                    cov = extra_cov_factor[z] * cov
                else:
                    cov = extra_cov_factor * cov

            if extra_diag_errors is not None:
                if type(extra_diag_errors) == tuple:
                    extra_diag_error = np.interp(
                        k, extra_diag_errors[0][z], extra_diag_errors[1][z]
                    )
                else:
                    extra_diag_error = extra_diag_errors[z][mask]
                di = np.diag_indices(cov.shape[0])
                cov[di] += extra_diag_error**2

            chi2_z = vector.dot(np.linalg.inv(cov).dot(vector))
            if return_redshift_chi2:
                chi2_list[i].append(chi2_z)
                ndof_list[i].append(vector.size)
            chi2 += chi2_z
            ndof += vector.size
        if not return_redshift_chi2:
            chi2_list.append(chi2)
            ndof_list.append(ndof)

    return chi2_list, ndof_list, zbin


def getChi2Pdf(dof, sigma=3):
    s = np.sqrt(2 * dof)
    x = np.linspace(dof - sigma * s, dof + sigma * s, 100)
    from scipy.stats import chi2 as chi2_stats

    rv = chi2_stats(dof)

    return x, rv.pdf(x)


def plot_chi2(ndofs, chi2s, chi2s_ref=None, save=None, fit_gausian=False, nbins=5):
    kstest = scipy.stats.ks_1samp(
        chi2s, scipy.stats.chi2(ndofs[0]).cdf, alternative="less"
    )
    kstest_two = scipy.stats.ks_1samp(chi2s, scipy.stats.chi2(ndofs[0]).cdf)
    print("One sided KS test p-value: ", kstest.pvalue)
    print("Two sided KS test p-value: ", kstest_two.pvalue)

    n, bins, patches = plt.hist(
        chi2s, bins=nbins, density=True, alpha=0.7, label=r"$p$=" f"{kstest.pvalue:.2f}"
    )

    if chi2s_ref is not None:
        plt.hist(
            chi2s_ref,
            bins=nbins,
            density=True,
            alpha=0.7,
            edgecolor="k",
            facecolor="None",
            label="Not corrected",
        )

    x, f_x = getChi2Pdf(ndofs[0])
    plt.plot(x, f_x, "k-", label=f"Expected ({int(ndofs[0]):d} dof)")

    if fit_gausian:
        bin_center = (bins[1:] + bins[0:-1]) / 2
        fit_function = lambda x, mu, sigma: np.exp(
            -1.0 * (x - mu) ** 2 / (2 * sigma**2)
        ) / np.sqrt(2 * np.pi * sigma**2)

        popt, pcov = curve_fit(
            fit_function, xdata=bin_center, ydata=n, p0=[np.mean(chi2s), np.std(chi2s)]
        )
        print("Corr with mean = ", np.mean(chi2s) / x[np.argmax(f_x)])
        print("Corr with fit gauss = ", popt[0] / x[np.argmax(f_x)])

        x_gauss = np.linspace(np.min(chi2s), np.max(chi2s), 100)
        plt.plot(
            x_gauss,
            fit_function(x_gauss, np.mean(chi2s), np.std(chi2s)),
            "k--",
            lw=2,
            label="fitted pdf",
        )
        plt.plot(
            x_gauss, fit_function(x_gauss, *popt), "r--", lw=2, label="fitted pdf 2"
        )

    plt.legend()

    ax = plt.gca()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    plt.ylabel("PDF")
    plt.xlabel("Chi-squared")
    m = np.sqrt(2 * ndofs[0])
    if save is not None:
        plt.savefig(f"{save}.pdf")
        plt.savefig(f"{save}.png")

        if chi2s_ref is not None:
            np.savetxt(
                f"{save}.txt",
                np.transpose(np.stack([ndofs, chi2s, chi2s_ref])),
                header="NDOF & CHI2 & CHI2 NO CORRECTION",
            )
        else:
            np.savetxt(
                f"{save}.txt",
                np.transpose(np.stack([ndofs, chi2s])),
                header="NDOF & CHI2",
            )


def plot_chi2_z(ndofs, chi2s, zbin, figsize=(10, 10), nbins=5):
    fig = plt.figure(figsize=figsize)

    for i, z in enumerate(zbin):
        ax = fig.add_subplot(4, 4, i + 1)
        kstest = scipy.stats.ks_1samp(
            chi2s[i], scipy.stats.chi2(ndofs[i][0]).cdf, alternative="less"
        )
        kstest_two = scipy.stats.ks_1samp(chi2s[i], scipy.stats.chi2(ndofs[i][0]).cdf)
        ax.hist(
            chi2s[i],
            bins=nbins,
            density=True,
            alpha=0.7,
            label=r"$p$="
            + f"{kstest.pvalue:.2f} "
            + r"$p2s$="
            + f"{kstest_two.pvalue:.2f}",
        )
        x, f_x = getChi2Pdf(ndofs[i][0])
        plt.plot(x, f_x, "k-", label=f"Expected ({int(ndofs[i][0]):d} dof)")
        plt.legend()
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

        ax.set_ylabel("PDF")
        ax.set_xlabel("Chi-squared")
        ax.set_title(f"z = {z}, Corr = {(np.mean(chi2s[i])/x[np.argmax(f_x)]):.2f}")
    fig.tight_layout()
