from p1desi import plotpk, utils, pk_io
import pickle, scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.lines import Line2D





def model_SB1(k,A,gamma,B,b,k1,phi1,C,c,k2,phi2):
    power_law = A * k**(-gamma) + B * np.exp(-b*k) * np.cos(2*np.pi*((k-k1)/k1) + phi1) + C * np.exp(-c*k) * np.cos(2*np.pi*((k-k2)/k2) + phi2)
    return(power_law)

def model_SB2(k,A,gamma,B,b,k1,phi1):
    power_law = A * k**(-gamma) + B * np.exp(-b*k) * np.cos(2*np.pi*((k-k1)/k1) + phi1)
    return(power_law)

def model_SB1_indiv(param_mean,redshift,A,B,k):
    mean_p = model_SB1(k*(1+redshift),*param_mean)
    return (A * k + B) * mean_p


def model_SB1_indiv_kms(param_mean,A,B,k):
    mean_p = model_SB1(k,*param_mean)
    return (A * k + B) * mean_p



def fit_model_SB1(x,y,dy,
                  k1,k2,dk,
                  xmin=None,
                  xmax=None):

    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy  = x[mask], y[mask], dy[mask]
    if(xmin is None):
        xmin= x.min()
    if(xmax is None):
        xmax= x.max()
    mask = (x >= xmin) & (x <=xmax)
    param, cov = curve_fit(model_SB1,
                           xdata=x[mask],
                           ydata=y[mask],
                           sigma=dy[mask],
                           p0=[1.0,1.0,
                               0.02,1.0,k1,0.0,
                               0.01,1.0,k2,0.0],
                           bounds = ([-np.inf,-np.inf,
                                      0.0,0.0,k1-dk,-2*np.pi,
                                      0.0,0.0,k2-dk,-2*np.pi],
                                     [np.inf,np.inf,
                                      0.5,np.inf,k1+dk,2*np.pi,
                                      0.1,np.inf,k2+dk,2*np.pi]))
    return param, cov



def fit_model_SB2(x,y,dy,
                  k1,dk,
                  xmin=None,
                  xmax=None):
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy  = x[mask], y[mask], dy[mask]
    if(xmin is None):
        xmin= x.min()
    if(xmax is None):
        xmax= x.max()
    mask = (x >= xmin) & (x <=xmax)
    param, cov = curve_fit(model_SB2,
                           xdata=x[mask],
                           ydata=y[mask],
                           sigma=dy[mask],
                           p0=[1.0,1.0,
                               0.02,1.0,k1,0.0],
                           bounds = ([-np.inf,-np.inf,
                                      0.0,0.0,k1-dk,-2*np.pi],
                                     [np.inf,np.inf,
                                      0.5,np.inf,k1+dk,2*np.pi]))
    return param, cov


def fit_model_SB1_indiv(param_mean,
                        redshift,
                        x,y,dy,
                        xmin=None,
                        xmax=None,
                        velunits=False):
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(dy))
    x, y, dy  = x[mask], y[mask], dy[mask]
    if(xmin is None):
        xmin= x.min()
    if(xmax is None):
        xmax= x.max()
    mask = (x >= xmin) & (x <=xmax)
    def model_to_fit(k,A,B):
        if velunits :
            return model_SB1_indiv_kms(param_mean,A,B,k)
        else :
            return model_SB1_indiv(param_mean,redshift,A,B,k)
    param, cov = curve_fit(model_to_fit,
                           xdata=x[mask],
                           ydata=y[mask],
                           sigma=dy[mask],
                           p0=[0.0,0.0],
                           bounds = ([-np.inf,-np.inf],
                                     [np.inf,np.inf]))
    return param, cov





def init_side_band_power(pSB1_name,pSB2_name,zmax):
    pSB1 = pk_io.Pk.read_from_picca(pSB1_name)
    pSB2 = pk_io.Pk.read_from_picca(pSB2_name)

    velunits = pSB1.velunits
    
    kSB1,kSB2 = [],[]
    krestSB1,krestSB2 = [],[]
    pkrestSB1,pkrestSB2= [],[]
    errorpkrestSB1,errorpkrestSB2= [],[]

    for z in pSB1.zbin:
        if z < zmax:
            krestSB1.append(pSB1.k[z]*(1+z))
            krestSB2.append(pSB2.k[z]*(1+z))
            kSB1.append(pSB1.k[z])
            kSB2.append(pSB2.k[z])

    if velunits :
        krestSB1 = np.array(kSB1)
        krestSB2 = np.array(kSB2)
        krescaleSB1 = np.linspace(np.nanmin(krestSB1),np.nanmax(krestSB1),krestSB1.shape[1])
        krescaleSB2 = np.linspace(np.nanmin(krestSB2),np.nanmax(krestSB2),krestSB2.shape[1])
    else :
        krestSB1 = np.array(krestSB1)
        krestSB2 = np.array(krestSB2)
        krescaleSB1 = np.linspace(krestSB1[-1][0],krestSB1[0][-1],krestSB1.shape[1])
        krescaleSB2 = np.linspace(krestSB2[-1][0],krestSB2[0][-1],krestSB2.shape[1])




    for i, z in enumerate(pSB1.zbin):
        if z < zmax:
            pkrestSB1.append(np.interp(krescaleSB1, krestSB1[i], pSB1.p[z]))
            errorpkrestSB1.append(np.interp(krescaleSB1, krestSB1[i], pSB1.err[z]))
            pkrestSB2.append(np.interp(krescaleSB2, krestSB2[i], pSB2.p[z]))
            errorpkrestSB2.append(np.interp(krescaleSB2, krestSB2[i], pSB2.err[z]))


    mean_dict = {}
    mean_dict["krescaleSB1"] = krescaleSB1
    mean_dict["krescaleSB2"] = krescaleSB2
    mean_dict["pkrestSB1"] = np.nanmean(pkrestSB1,axis=0)
    mean_dict["pkrestSB2"] = np.nanmean(pkrestSB2,axis=0)
    mean_dict["errorpkrestSB1"] = np.nanmean(errorpkrestSB1,axis=0)/np.sqrt(len(pkrestSB1))
    mean_dict["errorpkrestSB2"] = np.nanmean(errorpkrestSB2,axis=0)/np.sqrt(len(pkrestSB2))

    return pSB1, mean_dict


def fit_mean_side_band_rest(mean_dict,
                            dkrest = 1.0,
                            kminrest = None,
                            kmaxrest = 8,
                            velunits = False):

    if velunits :
        kc4 = utils.kc4_speed
        ksi4 = utils.ksi4_speed
    else :
        kc4 = utils.kc4
        ksi4 = utils.ksi4

    (param_SB1_mean,
     cov_SB1_mean) = fit_model_SB1(mean_dict["krescaleSB1"],
                                   mean_dict["pkrestSB1"],
                                   mean_dict["errorpkrestSB1"],
                                   kc4,
                                   ksi4,
                                   dkrest,
                                   xmin=kminrest,
                                   xmax=kmaxrest)

    (param_SB2_mean,
     cov_SB2_mean) = fit_model_SB2(mean_dict["krescaleSB2"],
                                   mean_dict["pkrestSB2"],
                                   mean_dict["errorpkrestSB2"],
                                   kc4,
                                   dkrest,
                                   xmin=kminrest,
                                   xmax=kmaxrest)
    return param_SB1_mean, param_SB2_mean, cov_SB1_mean, cov_SB2_mean



def fit_indiv_side_band_1(pSB1,
                          zmax,
                          param_SB1_mean,
                          kmin = None,
                          kmax = 8,
                          save_fit = None):
    param_SB1_indiv = []
    cov_SB1_indiv = []
    for z in pSB1.zbin:
        if z < zmax:
            (param_SB1,
             cov_SB1) = fit_model_SB1_indiv(param_SB1_mean,
                                            z,
                                            pSB1.k[z],
                                            pSB1.p[z],
                                            pSB1.err[z],
                                            xmin=kmin,
                                            xmax=kmax,
                                            velunits=pSB1.velunits)
            param_SB1_indiv.append(param_SB1)
            cov_SB1_indiv.append(cov_SB1)
    if save_fit is not None:
        pickle.dump((param_SB1_mean,param_SB1_indiv),open(save_fit,"wb"))
    return param_SB1_indiv, cov_SB1_indiv


def plot_side_band_fit(name_out,
                       plot_P,
                       mean_dict,
                       pSB1,
                       zmax,
                       param_SB1_mean,
                       param_SB2_mean,
                       param_SB1_indiv,
                       nb_bins,
                       kmaxrest,
                       kmax,
                       **kwargs):

    style = utils.return_key(kwargs,"style",None)
    if(style is not None):
        plt.style.use(style)

    figsize = utils.return_key(kwargs,"figsize", (9,9))
    ncol_legend = utils.return_key(kwargs,"ncol_legend", 2)
    size_legend = utils.return_key(kwargs,"size_legend", 11)
    fontsize = utils.return_key(kwargs,"fontsize", 15)
    size = utils.return_key(kwargs,"size", 14)
    markersize = utils.return_key(kwargs,"markersize", 8)

    ylim = utils.return_key(kwargs,"ylim", [-0.01,0.05] if plot_P else [-0.001,0.02])
    ylabel_1 = utils.return_key(kwargs,"ylabel_1", r'$P_{\mathrm{SB}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$' if pSB1.velunits else r'$P_{\mathrm{SB}}~[\AA]$')
    ylabel_2 = utils.return_key(kwargs,"ylabel_2", r'$P_{\mathrm{SB1}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$' if pSB1.velunits else r'$P_{\mathrm{SB1}}~[\AA]$')
    xlabel_1 = utils.return_key(kwargs,"xlabel_1", r'$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$' if pSB1.velunits else r'$k_{\mathrm{rest}}=k_{\mathrm{obs}}(1+z)~[\mathrm{\AA}^{-1}]$')
    xlabel_2 = utils.return_key(kwargs,"xlabel_2", r'$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$' if pSB1.velunits else r'$k_{\mathrm{obs}}~[\mathrm{\AA}^{-1}]$')

    fig,ax=plt.subplots(2,1,figsize=figsize,sharex=False)

    if kmaxrest is None:
        kmaxrest = np.max(mean_dict["krescaleSB1"])

    k_fit_mean_rest = np.linspace(np.min(mean_dict["krescaleSB1"]),kmaxrest,nb_bins)
    if(plot_P):
        p_plot_SB1 = mean_dict["pkrestSB1"]
        err_plot_SB1 = mean_dict["errorpkrestSB1"]
        p_plot_fit_SB1 = model_SB1(k_fit_mean_rest,*param_SB1_mean)
        p_plot_SB2 = mean_dict["pkrestSB2"]
        err_plot_SB2 = mean_dict["errorpkrestSB2"]
        p_plot_fit_SB2 = model_SB2(k_fit_mean_rest,*param_SB2_mean)
    else:
        p_plot_SB1 = mean_dict["krescaleSB1"] * mean_dict["pkrestSB1"] / np.pi
        err_plot_SB1 = mean_dict["krescaleSB1"] * mean_dict["errorpkrestSB1"] / np.pi
        p_plot_fit_SB1 = k_fit_mean_rest * model_SB1(k_fit_mean_rest,*param_SB1_mean) / np.pi
        p_plot_SB2 = mean_dict["krescaleSB2"] * mean_dict["pkrestSB2"] / np.pi
        err_plot_SB2 = mean_dict["krescaleSB2"] * mean_dict["errorpkrestSB2"] / np.pi
        p_plot_fit_SB2 = k_fit_mean_rest * model_SB2(k_fit_mean_rest,*param_SB2_mean) / np.pi

    ax[0].errorbar(mean_dict["krescaleSB1"],
                   p_plot_SB1,
                   err_plot_SB1,
                   marker=".",
                   ls='None',
                   markersize=markersize)
    ax[0].plot(k_fit_mean_rest,
               p_plot_fit_SB1,
               color="C0")

    ax[0].errorbar(mean_dict["krescaleSB2"],
                   p_plot_SB2,
                   err_plot_SB2,
                   marker=".",
                   ls='None',
                   markersize=markersize)
    ax[0].plot(k_fit_mean_rest,
               p_plot_fit_SB2,
               color="C1")

    color = cm.rainbow(np.linspace(0, 1, len(pSB1.zbin[pSB1.zbin < zmax])))


    for i, z in enumerate(pSB1.zbin):
        if z < zmax:
            k_fit = np.linspace(np.nanmin(pSB1.k[z]),kmax,nb_bins)
            if pSB1.velunits :
                model_fitted = model_SB1_indiv_kms(param_SB1_mean,
                                                param_SB1_indiv[i][0],
                                                param_SB1_indiv[i][1],
                                                k_fit)
            else :
                model_fitted = model_SB1_indiv(param_SB1_mean,
                                            z,
                                            param_SB1_indiv[i][0],
                                            param_SB1_indiv[i][1],
                                            k_fit)
            if plot_P:
                p_plot = pSB1.p[z]
                err_plot = pSB1.err[z]
                p_plot_fit = model_fitted
            else:
                p_plot = pSB1.norm_p[z]
                err_plot = pSB1.norm_err[z]
                p_plot_fit = k_fit * model_fitted / np.pi
            ax[1].errorbar(pSB1.k[z],
                        p_plot,
                        err_plot,
                        label=f'z = {z:.1f} ({pSB1.number_chunks[z]} chunks)',
                        marker=".",
                        ls='None',
                        color=color[i],
                        markersize=markersize)
            ax[1].plot(k_fit,
                    p_plot_fit,
                    color=color[i])

    ax[1].set_ylim(ylim)
    ax[0].set_ylim(ylim)
    ax[1].set_xlim([0,kmax])
    ax[0].set_xlim([0,kmaxrest])


    ax[0].set_ylabel(ylabel_1,fontsize=fontsize)
    ax[0].set_xlabel(xlabel_1,fontsize=fontsize)
    ax[0].tick_params("x",labelsize=size)
    ax[0].tick_params("y",labelsize=size)

    ax[1].legend(ncol=ncol_legend,fontsize=size_legend)
    ax[1].set_ylabel(ylabel_2,fontsize=fontsize)
    ax[1].set_xlabel(xlabel_2,fontsize=fontsize)
    ax[1].tick_params("x",labelsize=size)
    ax[1].tick_params("y",labelsize=size)


    legend_elements = [Line2D([], [], color='C0', marker=None, linestyle='-', label='SB1'),
                       Line2D([], [], color='C1', marker=None, linestyle='-', label='SB2')]
    ax[0].legend(handles=legend_elements,fontsize=size_legend)

    plt.tight_layout()
    plt.savefig(name_out)





def fit_and_plot_side_band(pSB1_name,
                           pSB2_name,
                           name_out,
                           zmax,
                           nb_bins,
                           dkrest,
                           kminrest,
                           kmaxrest,
                           kmin,
                           kmax,
                           plot_P = True,
                           save_fit = None,
                           **plt_args):

    pSB1, mean_dict = init_side_band_power(pSB1_name,pSB2_name,zmax)

    (param_SB1_mean,
     param_SB2_mean,
     cov_SB1_mean,
     cov_SB2_mean) = fit_mean_side_band_rest(mean_dict,
                                             dkrest = dkrest,
                                             kminrest = kminrest,
                                             kmaxrest = kmaxrest,
                                             velunits = pSB1.velunits)

    (param_SB1_indiv,
     cov_SB1_indiv) = fit_indiv_side_band_1(pSB1,
                                            zmax,
                                            param_SB1_mean,
                                            kmin = kmin,
                                            kmax = kmax,
                                            save_fit = save_fit)

    plot_side_band_fit(name_out,
                       plot_P,
                       mean_dict,
                       pSB1,
                       zmax,
                       param_SB1_mean,
                       param_SB2_mean,
                       param_SB1_indiv,
                       nb_bins,
                       kmaxrest,
                       kmax,
                       **plt_args)
    return (param_SB1_mean,
            param_SB2_mean,
            cov_SB1_mean,
            cov_SB2_mean,
            param_SB1_indiv,
            cov_SB1_indiv)
