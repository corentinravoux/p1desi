import pickle
import matplotlib.pyplot as plt
from p1desi import utils


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


def plot_stat_uncertainties(mean_pk, outname, zmax, **plot_args):

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

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    for z in mean_pk.zbin:
        if z < zmax:

            if mean_pk.velunits:
                kmax = float(utils.kAAtokskm(kmax_AA, z=z))
                kmin = float(utils.kAAtokskm(kmin_AA, z=z))
            else:
                kmax = kmax_AA
                kmin = kmin_AA

            mask = (mean_pk.k[z] > kmin) & (mean_pk.k[z] < kmax)
            ax[0].semilogy(
                mean_pk.k[z][mask],
                mean_pk.err[z][mask],
                label=r"$z = ${:1.1f}".format(z),
            )
            ax[1].semilogy(
                mean_pk.k[z][mask],
                mean_pk.err[z][mask] / mean_pk.p[z][mask],
                label=r"$z = ${:1.1f}".format(z),
            )
    if mean_pk.velunits:
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


""" 
def plot_syst_uncertainties(mean_pk, outname, zmax, **plot_args):


    fig, ax = plt.subplots(7,2, figsize = (11,13),sharex=True)
    ax[0][0].set_xlim(kmin,kmax)


    syste_tot = [[] for i in range(len(mean_pk.zbin[mean_pk.zbin<zmax]))]

    for iz,z in enumerate(mean_pk.zbin):
        if z < zmax:
            syste_noise = 0.3*(mean_pk.p_noise_miss[z]/mean_pk.resocor[z])
            syste_tot[iz].append(syste_noise**2)
            ax[0][1].plot(mean_pk.k[z],syste_noise/mean_pk.err[z])
            ax[0][0].plot(mean_pk.k[z],syste_noise)
            ax[0][0].set_title("Noise estimation", x=title_shift, y=title_yshift,fontsize = title_size)

            syste_reso = 2 * mean_pk.k[z]**2 * delta_l[iz] * delta_delta_l[iz] * mean_pk.p[z]
            syste_tot[iz].append(syste_reso**2)
            ax[1][1].plot(mean_pk.k[z],syste_reso/mean_pk.err[z])
            ax[1][0].plot(mean_pk.k[z],syste_reso)
            ax[1][0].set_title("Resolution", x=title_shift, y=title_yshift,fontsize = title_size)

            syste_sb = 0.3 * pk_sb.err[z]
            syste_tot[iz].append(syste_sb**2)
            ax[2][1].plot(mean_pk.k[z],syste_sb/mean_pk.err[z])
            ax[2][0].plot(mean_pk.k[z],syste_sb)
            ax[2][0].set_title("Side band", x=title_shift, y=title_yshift,fontsize = title_size)

            syste_lines = 0.3 * np.abs(np.poly1d(lines_coeff_fit[iz])(mean_pk.k[z])-1) * mean_pk.p[z]
            syste_tot[iz].append(syste_lines**2)
            ax[3][1].plot(mean_pk.k[z],syste_lines/mean_pk.err[z],label =r'$z = ${:1.1f}'.format(z))
            ax[3][0].plot(mean_pk.k[z],syste_lines)
            ax[3][0].set_title("Line masking", x=title_shift, y=title_yshift,fontsize = title_size)

            syste_hcd = 0.3 * np.abs(hcd_coeff_fit[iz]-1) * mean_pk.p[z]
            syste_tot[iz].append(syste_hcd**2)
            ax[4][1].plot(mean_pk.k[z],syste_hcd/mean_pk.err[z])
            ax[4][0].plot(mean_pk.k[z],syste_hcd)
            ax[4][0].set_title("DLA masking", x=title_shift, y=title_yshift,fontsize = title_size)

            syste_residual = 0.3 * np.abs(corrections.model_cont_correction(mean_pk.k[z],*residual_coeff_fit[iz])-1) * mean_pk.p[z]
            syste_tot[iz].append(syste_residual**2)
            ax[5][1].plot(mean_pk.k[z],syste_residual/mean_pk.err[z])
            ax[5][0].plot(mean_pk.k[z],syste_residual)
            ax[5][0].set_title("Residual correction", x=title_shift, y=title_yshift,fontsize = title_size)

            
            A_dla_completeness = hcd.rogers(z,mean_pk.k[z],*dla_completeness_coef[iz])  
            syste_dla_completeness = 0.2 * np.abs(A_dla_completeness - 1) * mean_pk.p[z]        
            syste_tot[iz].append(syste_dla_completeness**2)
            ax[6][1].plot(mean_pk.k[z],syste_dla_completeness/mean_pk.err[z])
            ax[6][0].plot(mean_pk.k[z],syste_dla_completeness)
            ax[6][0].set_title("DLA completeness", x=title_shift, y=title_yshift,fontsize = title_size)

        
    for iz,z in enumerate(syste_tot):
        syste_tot[iz] = np.sqrt(np.sum(syste_tot[iz],axis=0))

        
    handles, labels = ax[0][0].get_legend_handles_labels()
    ax[3][1].legend(loc=2, bbox_to_anchor=(1.04, 1.25), borderaxespad=0.,fontsize = fontlegend)


    for i in range(7):
        ax[i][1].set_ylim(bottom=0.0)
        ax[i][1].set_ylabel(r"$\sigma_{\mathrm{syst}} / \sigma_{\mathrm{stat}}$",fontsize = fontsize_y)
        ax[i][0].set_ylabel(r"$\sigma_{\mathrm{syst}}$",fontsize = fontsize_y)
        ax[i][0].set_yscale("log")
        ax[i][0].yaxis.set_tick_params(labelsize=size)
        ax[i][1].yaxis.set_tick_params(labelsize=size)



    ax[0][1].set_ylim(top = 1.0)
    ax[2][1].set_ylim(top = 0.3)
    ax[3][1].set_ylim(top = 1.5)
    ax[5][1].set_ylim(top = 0.2)
    ax[6][1].set_ylim(top = 1.0)

    ax[1][0].set_ylim(bottom=0.0001)
    ax[4][0].set_ylim(bottom=0.00002)
    ax[5][0].set_ylim(bottom=0.00002)
    ax[6][0].set_ylim(bottom=0.0001)

    ax[0][0].set_ylim(top = 0.006)
    ax[1][0].set_ylim(top = 0.02)
    ax[2][0].set_ylim(top = 0.006)
    ax[3][0].set_ylim(top = 0.04)
    ax[4][0].set_ylim(top = 0.006)
    ax[5][0].set_ylim(top = 0.02)
    ax[6][0].set_ylim(top = 0.03)



    ax[-1][0].set_xlabel(r'$k~[\mathrm{\AA}^{-1}]$',fontsize = fontsize)
    ax[-1][0].xaxis.set_tick_params(labelsize=size)
    ax[-1][1].set_xlabel(r'$k~[\mathrm{\AA}^{-1}]$',fontsize = fontsize)
    ax[-1][1].xaxis.set_tick_params(labelsize=size)


    fig.subplots_adjust(wspace = 0.2,hspace = 0.45,right=0.85)

    plt.savefig(f"out/systematics.pdf")
    pickle.dump(syste_tot,open("out/sytematics_total.pickle","wb"))    
 """
