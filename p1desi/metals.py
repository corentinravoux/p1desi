from p1desi import plotpk, utils
import pickle, scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.lines import Line2D


def plot_side_band_study(zbins,
                         data,
                         out_name,
                         mean_dict,
                         noise_to_plot,
                         labelnoise,
                         k_units,
                         side_band_legend,
                         side_band_comp = None,
                         side_band_fitpolynome = False,
                         **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)
    fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)

    for z,d in zip(zbins,data):
        ax[0].plot(d['meank'],d['meanPk_raw'],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[0].set_ylabel('$P_{raw} [\AA]$')
        elif(k_units == "kms"):
            ax[0].set_ylabel('$P_{raw} [km/s]$')
        ax[0].legend()
        ax[1].plot(d['meank'],d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [\AA]$')
        elif(k_units == "kms"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [km/s]$')
        ax[2].plot(d['meank'],d['meanPk_raw'] - d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[2].set_ylabel('$ (P_{raw} - P_{pipeline}) [\AA]$')
        elif(k_units == "kms"):
            ax[2].set_ylabel('$ (P_{raw} - P_{pipeline}) [km/s]$')

    ax[3].errorbar(mean_dict["k_array"],mean_dict["meanPk"],mean_dict["errorPk"], fmt = 'o',label=side_band_legend[0])
    if(k_units == "A"):
        ax[3].set_ylabel('$mean_{z}(P_{SB}) [\AA]$')
    elif(k_units == "kms"):
        ax[3].set_ylabel('$mean_{z}(P_{SB}) [km/s]$')
    if(side_band_fitpolynome):
        poly = scipy.polyfit(mean_dict["k_array"],mean_dict["meanPk"],6)
        Poly = np.polynomial.polynomial.Polynomial(np.flip(poly))
        cont_k_array = np.linspace(np.min(mean_dict["k_array"]),np.max(mean_dict["k_array"]),300)
        polynome = Poly(cont_k_array)
        mean_dict["poly"]= polynome
        mean_dict["k_cont"]=cont_k_array
        ax[3].plot(cont_k_array,polynome)
    if(side_band_comp is not None):
        yerr =np.sqrt( side_band_comp["error_meanPk_noise"]**2 + side_band_comp["error_meanPk_raw"]**2)
        ax[3].errorbar(side_band_comp["k_array"],side_band_comp["meanPk_raw"] - side_band_comp["meanPk_noise"],yerr, fmt = 'o',label=side_band_legend[1])
        if(side_band_fitpolynome):
            ax[3].plot(side_band_comp["k_cont"],side_band_comp["poly"])
        ax[3].legend()
    if(k_units == "A"):
        ax[3].set_xlabel('k[1/$\AA$]')
        utils.place_k_speed_unit_axis(fig,ax[0])
    elif(k_units == "kms"):
        ax[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax[0].set_xlim(kmin,kmax)
    fig.tight_layout()
    fig.savefig(f"{out_name}_side_band_unit{k_units}.pdf",format="pdf")






def plot_metal_study(data,
                     zbins,
                     out_name,
                     k_units,
                     use_diff_noise,
                     plot_side_band,
                     side_band_comp=None,
                     side_band_legend=["SB1","SB2"],
                     **kwargs):

    mean_dict = return_mean_z_dict(zbins,data)
    if(use_diff_noise):
        noise_to_plot,labelnoise = 'meanPk_diff','diff'
    else:
        noise_to_plot,labelnoise = 'meanPk_noise','pipeline'

    if(plot_side_band):
        plot_side_band_study(zbins,
                             data,
                             out_name,
                             mean_dict,
                             noise_to_plot,
                             labelnoise,
                             k_units,
                             side_band_legend,
                             side_band_comp=side_band_comp,
                             **kwargs)







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



def fit_model_SB2(nb_bins,
                  x,y,dy,
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





def init_side_band_power(pSB1_name,pSB2_name,zbins,velunits=False):

    pSB1 = plotpk.read_pk_means(pSB1_name)
    pSB2 = plotpk.read_pk_means(pSB2_name)

    kSB1,kSB2 = [], []
    pkSB1,pkSB2 = [], []
    errorpkSB1,errorpkSB2 = [], []

    krestSB1,krestSB2 = [], []
    pkrestSB1,pkrestSB2=[],[]
    errorpkrestSB1,errorpkrestSB2=[],[]

    nSB1,nSB2 = [],[]


    for z,d in zip(zbins,pSB1):
        pkSB1.append(np.array(d['meanPk']))
        errorpkSB1.append(np.array(d['errorPk']))
        kSB1.append(np.array(d['meank']))
        krestSB1.append(np.array(d['meank'])*(1+z))
        nSB1.append(d["N_chunks"])

    for z,d in zip(zbins,pSB2):
        pkSB2.append(np.array(d['meanPk']))
        errorpkSB2.append(np.array(d['errorPk']))
        kSB2.append(np.array(d['meank']))
        krestSB2.append(np.array(d['meank'])*(1+z))
        nSB2.append(d["N_chunks"])

    dict_redshift = {}

    dict_redshift["kSB1"] = np.array(kSB1)
    dict_redshift["pkSB1"] = np.array(pkSB1)
    dict_redshift["errorpkSB1"] = np.array(errorpkSB1)
    dict_redshift["nSB1"] = np.array(nSB1)
    dict_redshift["kSB2"] = np.array(kSB2)
    dict_redshift["pkSB2"] = np.array(pkSB2)
    dict_redshift["errorpkSB2"] = np.array(errorpkSB2)
    dict_redshift["nSB2"] = np.array(nSB2)

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

    for i in range(len(pkSB1)):
        pkrestSB1.append(np.interp(krescaleSB1, krestSB1[i], pkSB1[i]))
        errorpkrestSB1.append(np.interp(krescaleSB1, krestSB1[i], errorpkSB1[i]))
        pkrestSB2.append(np.interp(krescaleSB2, krestSB2[i], pkSB2[i]))
        errorpkrestSB2.append(np.interp(krescaleSB2, krestSB2[i], errorpkSB2[i]))


    mean_dict = {}
    mean_dict["krescaleSB1"] = krescaleSB1
    mean_dict["krescaleSB2"] = krescaleSB2
    mean_dict["pkrestSB1"] = np.nanmean(pkrestSB1,axis=0)
    mean_dict["pkrestSB2"] = np.nanmean(pkrestSB2,axis=0)
    mean_dict["errorpkrestSB1"] = np.nanmean(errorpkrestSB1,axis=0)/np.sqrt(len(pkrestSB1))
    mean_dict["errorpkrestSB2"] = np.nanmean(errorpkrestSB2,axis=0)/np.sqrt(len(pkrestSB2))

    return dict_redshift, mean_dict


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



def fit_indiv_side_band_1(dict_redshift,
                          zbins,
                          param_SB1_mean,
                          nb_bins=1000,
                          kmin = None,
                          kmax = 8,
                          save_fit = None,
                          velunits=False):
    param_SB1_indiv = []
    cov_SB1_indiv = []
    for i in range(len(zbins)):
        (param_SB1,
         cov_SB1) = fit_model_SB1_indiv(param_SB1_mean,
                                        zbins[i],
                                        dict_redshift["kSB1"][i],
                                        dict_redshift["pkSB1"][i],
                                        dict_redshift["errorpkSB1"][i],
                                        xmin=kmin,
                                        xmax=kmax,
                                        velunits=velunits)
        param_SB1_indiv.append(param_SB1)
        cov_SB1_indiv.append(cov_SB1)
    if save_fit is not None:
        pickle.dump((param_SB1_mean,param_SB1_indiv),open(save_fit,"wb"))
    return param_SB1_indiv, cov_SB1_indiv


def plot_side_band_fit(name_out,
                       plot_P,
                       mean_dict,
                       dict_redshift,
                       zbins,
                       param_SB1_mean,
                       param_SB2_mean,
                       param_SB1_indiv,
                       nb_bins,
                       kmaxrest,
                       kmax,
                       velunits=False,
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
    ylabel_1 = utils.return_key(kwargs,"ylabel_1", r'$P_{\mathrm{SB}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$' if velunits else r'$P_{\mathrm{SB}}~[\AA]$')
    ylabel_2 = utils.return_key(kwargs,"ylabel_2", r'$P_{\mathrm{SB1}}~[\mathrm{km}\cdot\mathrm{s}^{-1}]$' if velunits else r'$P_{\mathrm{SB1}}~[\AA]$')
    xlabel_1 = utils.return_key(kwargs,"xlabel_1", r'$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$' if velunits else r'$k_{\mathrm{rest}}=k_{\mathrm{obs}}(1+z)~[\mathrm{\AA}^{-1}]$')
    xlabel_2 = utils.return_key(kwargs,"xlabel_2", r'$k~[\mathrm{s}\cdot\mathrm{km}^{-1}]$' if velunits else r'$k_{\mathrm{obs}}~[\mathrm{\AA}^{-1}]$')

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

    color = cm.rainbow(np.linspace(0, 1, len(zbins)))


    for i in range(len(zbins)):

        k_fit = np.linspace(np.nanmin(dict_redshift["kSB1"][i]),kmax,nb_bins)
        if velunits :
            model_fitted = model_SB1_indiv_kms(param_SB1_mean,
                                               param_SB1_indiv[i][0],
                                               param_SB1_indiv[i][1],
                                               k_fit)
        else :
            model_fitted = model_SB1_indiv(param_SB1_mean,
                                           zbins[i],
                                           param_SB1_indiv[i][0],
                                           param_SB1_indiv[i][1],
                                           k_fit)
        if plot_P:
            p_plot = dict_redshift["pkSB1"][i]
            err_plot = dict_redshift["errorpkSB1"][i]
            p_plot_fit = model_fitted
        else:
            p_plot = dict_redshift["kSB1"][i] * dict_redshift["pkSB1"][i] / np.pi
            err_plot = dict_redshift["kSB1"][i] * dict_redshift["errorpkSB1"][i] / np.pi
            p_plot_fit = k_fit * model_fitted / np.pi
        ax[1].errorbar(dict_redshift["kSB1"][i],
                       p_plot,
                       err_plot,
                       label=f'z = {zbins[i]:.1f} ({dict_redshift["nSB1"][i]} chunks)',
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
                           zbins,
                           nb_bins,
                           dkrest,
                           kminrest,
                           kmaxrest,
                           kmin,
                           kmax,
                           plot_P = True,
                           save_fit = None,
                           velunits = False,
                           **plt_args):

    dict_redshift, mean_dict = init_side_band_power(pSB1_name,pSB2_name,zbins,velunits=velunits)

    (param_SB1_mean,
     param_SB2_mean,
     cov_SB1_mean,
     cov_SB2_mean) = fit_mean_side_band_rest(mean_dict,
                                             dkrest = dkrest,
                                             kminrest = kminrest,
                                             kmaxrest = kmaxrest,
                                             velunits = velunits)

    (param_SB1_indiv,
     cov_SB1_indiv) = fit_indiv_side_band_1(dict_redshift,
                                            zbins,
                                            param_SB1_mean,
                                            nb_bins=nb_bins,
                                            kmin = kmin,
                                            kmax = kmax,
                                            save_fit = save_fit,
                                            velunits=velunits)

    plot_side_band_fit(name_out,
                       plot_P,
                       mean_dict,
                       dict_redshift,
                       zbins,
                       param_SB1_mean,
                       param_SB2_mean,
                       param_SB1_indiv,
                       nb_bins,
                       kmaxrest,
                       kmax,
                       velunits=velunits,
                       **plt_args)
    return (param_SB1_mean,
            param_SB2_mean,
            cov_SB1_mean,
            cov_SB2_mean,
            param_SB1_indiv,
            cov_SB1_indiv)
