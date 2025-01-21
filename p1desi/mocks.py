import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sgolay2


def compute_chi2(
    mean_pk_list,
    mean_pk_ref,
    kmin=0.0,
    kmax=2.0,
    zmin=2.1,
    zmax=4.7,
    use_boot=False,
    extra_diag_errors=None,
    use_only_diag=False,
    smooth_cov=True,
    smooth_cov_window=15,
    smooth_cov_order=5,
):

    chi2_list, ndof_list = [], []

    for mean_pk in mean_pk_list:
        zbin = mean_pk.zbin
        chi2 = 0.0
        ndof = 0
        w = (zbin > zmin) & (zbin < zmax)
        zbin = zbin[w]

        for i, z in enumerate(zbin):

            if use_only_diag:
                mask = (mean_pk.k[z] > kmin) & (mean_pk.k[z] < kmax)

                diag = mean_pk.err[z][mask] ** 2
                if extra_diag_errors is not None:
                    diag += extra_diag_errors[z][mask]
                cov = np.diag(diag)
                vector = mean_pk.p[z][mask] - mean_pk_ref.p[z][mask]

            else:
                if extra_diag_errors is not None:
                    if use_boot:
                        cov = mean_pk.boot_cov[z].copy()
                    else:
                        cov = mean_pk.cov[z].copy()
                    di = np.diag_indices(cov.shape[0])
                    cov[di] += extra_diag_errors[z]
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
            if smooth_cov:
                diag = np.copy(np.diag(cov))
                np.fill_diagonal(cov, np.full_like(diag, 0.0))
                cov = sgolay2.SGolayFilter2(
                    window_size=smooth_cov_window, poly_order=smooth_cov_order
                )(cov)
                np.fill_diagonal(cov, diag)

            chi2_z = vector.dot(np.linalg.inv(cov).dot(vector))
            chi2 += chi2_z
            ndof += vector.size
        chi2_list.append(chi2)
        ndof_list.append(ndof)

    return chi2_list, ndof_list


def getChi2Pdf(dof, sigma=3):
    s = np.sqrt(2 * dof)
    x = np.linspace(dof - sigma * s, dof + sigma * s, 100)
    from scipy.stats import chi2 as chi2_stats

    rv = chi2_stats(dof)

    return x, rv.pdf(x)


def plot_chi2(ndofs, chi2s, chi2s_ref=None, save=None):
    kstest = scipy.stats.ks_1samp(
        chi2s, scipy.stats.chi2(ndofs[0]).cdf, alternative="less"
    )
    kstest_two = scipy.stats.ks_1samp(chi2s, scipy.stats.chi2(ndofs[0]).cdf)
    print("One sided KS test p-value: ", kstest.pvalue)
    print("Two sided KS test p-value: ", kstest_two.pvalue)

    plt.hist(
        chi2s, bins=5, density=True, alpha=0.7, label=r"$p$=" f"{kstest.pvalue:.2f}"
    )
    if chi2s_ref is not None:
        plt.hist(
            chi2s_ref,
            bins=5,
            density=True,
            alpha=0.7,
            edgecolor="k",
            facecolor="None",
            label="Not corrected",
        )

    plt.plot(*getChi2Pdf(ndofs[0]), "k-", label=f"Expected ({int(ndofs[0]):d} dof)")
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
