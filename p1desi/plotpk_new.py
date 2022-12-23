#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import scipy
import astropy.table as t

from p1desi import utils



# CR - once all plotting functions in notebooks are adapted, add them here


def plot(pk,
         zmax,
         outname=None,
         plot_P=False,
         systematics_file = None,
         **kwargs):


    zbins = pk.zbin[pk.zbin < zmax]

    marker_size = utils.return_key(kwargs,"marker_size",7)
    marker_style = utils.return_key(kwargs,"marker_style",".")
    fontsize_x = utils.return_key(kwargs,"fontsize_x",16)
    fontsize_y = utils.return_key(kwargs,"fontsize_y",19)
    labelsize = utils.return_key(kwargs,"labelsize",14)
    fontlegend = utils.return_key(kwargs,"fontl",14)
    color = utils.return_key(kwargs,"color",[f"C{i}" for i in range(len(zbins))])
    kmin = utils.return_key(kwargs,"kmin",0.145)
    kmax = utils.return_key(kwargs,"kmax",2.5)
    ymin = utils.return_key(kwargs,"ymin",0.01)
    ymax = utils.return_key(kwargs,"ymax",0.2)
    figsize = utils.return_key(kwargs,"figsize",(11,8.5))
    place_velaxis = utils.return_key(kwargs,"place_velaxis",True)

    fig,ax = plt.subplots(1,figsize = figsize)

    if systematics_file is not None:
        systematics = utils.load_systematics_file(systematics_file)

    for i,z in enumerate(zbins):
        if systematics_file is not None:
            error_bar = np.sqrt(pk.err[z]**2 + systematics[i]**2)
        else:
            error_bar = pk.err[z]

        if plot_P:
            ax.errorbar(pk.k[z],
                        pk.p[z],
                        yerr = error_bar,
                        fmt = marker_style,
                        color = color[i],
                        markersize = marker_size,
                        label =r'$z = ${:1.1f}  ({} chunks)'.format(z,pk.number_chunks[z]))
        else:
            ax.errorbar(pk.k[z],
                        pk.norm_p[z],
                        yerr = pk.k[z] * error_bar / np.pi,
                        fmt = marker_style,
                        color = color[i],
                        markersize = marker_size,
                        label =r'$z = ${:1.1f}  ({} chunks)'.format(z,pk.number_chunks[z]))


    ax.set_xlabel(r'$k~[\mathrm{\AA}^{-1}]$', fontsize = fontsize_x)
    ax.set_ylabel(r'$\Delta_{1\mathrm{D},\alpha}^{2}$', fontsize=fontsize_y, labelpad=-1)

    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(direction='in')
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    ax.set_xlim(kmin,kmax)
    ax.set_ylim(ymin,ymax)


    ax.legend(loc=2, bbox_to_anchor=(1.03, 0.9), borderaxespad=0.,fontsize = fontlegend)
    fig.subplots_adjust(top=0.75,bottom=0.08,left=0.08,right=0.7,hspace=0.2,wspace=0.2)
    if pk.velunits is False:
        if place_velaxis:
            utils.place_k_speed_unit_axis(fig,ax,fontsize=fontsize_x,size=labelsize,pos=0.15)

    if outname is not None:
        fig.savefig(outname)




""" def plot_comparison(pk,
                    pk2,
                    zmax,
                    outname=None,
                    plot_P=False,
                    systematics_file = None,
                    **kwargs):


    zbins = pk.zbin[pk.zbin < zmax]

    apply_DESI_maskcont_corr = True
    apply_eBOSS_maskcont_corr = False
    apply_DESI_sb_corr = True
    apply_eBOSS_sb_corr = False
    correction_to_apply = ["lines","hcd"]


    dict_plot = corrections.apply_p1d_corections(mean_pk,
                                                zbins,
                                                plot_P,
                                                z_binsize,
                                                velunits,
                                                apply_DESI_maskcont_corr,
                                                apply_eBOSS_maskcont_corr,
                                                apply_DESI_sb_corr,
                                                apply_eBOSS_sb_corr,
                                                correction_to_apply,
                                                file_correction_hcd = file_correction_hcd,
                                                file_correction_hcd_eboss = file_correction_hcd_eboss,
                                                file_correction_lines = file_correction_lines,
                                                file_correction_lines_eboss = file_correction_lines_eboss,
                                                file_correction_cont = file_correction_cont,
                                                file_correction_cont_eboss = file_correction_cont_eboss,
                                                file_metal = file_metal,
                                                file_metal_eboss = file_metal_eboss)

    dr14 = "/global/cfs/cdirs/desi/users/ravouxco/pk1d/eboss_data/rerun_ebossdata_ebosspipeline/pk_DR12_V5_7_Final_all_ascii_nocorr.out"

    f_eb = np.loadtxt(dr14)


    fig,ax = plt.subplots(2,1,figsize = (10,8),gridspec_kw=dict(height_ratios=[3,1]),sharex=True)
    syste_tot = pickle.load(open("sytematics_total.pickle","rb"))    


    A_hcd_eboss = corrections.prepare_hcd_correction_eboss(zbins,file_correction_hcd_eboss)
    A_lines_eboss = corrections.prepare_lines_correction_eboss(zbins,file_correction_lines_eboss)
    A_cont_eboss = corrections.prepare_cont_correction_eboss(zbins,file_correction_cont_eboss)

    param_sb_eboss = corrections.prepare_metal_correction_eboss(file_metal_eboss)
    nbin_eboss=35
    klimInf_eboss=0.000813
    klimSup_eboss=klimInf_eboss + nbin_eboss*0.000542

    mean_error = []
    for iz,z in enumerate(zbins):
        mask = f_eb[:,0] == z
        
        k_eb = f_eb[:,1][mask] 
        
        
        
        mask_sb = param_sb_eboss[:,0] == z
        k_sb_eboss = param_sb_eboss[:,1][mask_sb]    
        sb_eboss = param_sb_eboss[:,2][mask_sb]    

        deltak_eboss = (klimSup_eboss-klimInf_eboss)/nbin_eboss
        ib_eboss =((k_eb-klimInf_eboss)/deltak_eboss).astype(int)
        cor_sb = np.zeros(k_eb.shape)
        slope_eboss = np.zeros(k_eb.shape)

        mask_k_sb_1 = k_eb < klimInf_eboss
        slope_eboss[mask_k_sb_1] = (sb_eboss[1]-sb_eboss[0])/deltak_eboss
        cor_sb[mask_k_sb_1] = sb_eboss[0] + slope_eboss[mask_k_sb_1] * (k_eb[mask_k_sb_1] - deltak_eboss/2 - klimInf_eboss)

        mask_k_sb_2 = (ib_eboss < nbin_eboss -1)&(~mask_k_sb_1)
        slope_eboss[mask_k_sb_2] = (sb_eboss[ib_eboss[mask_k_sb_2]+1] - sb_eboss[ib_eboss[mask_k_sb_2]])/deltak_eboss
        cor_sb[mask_k_sb_2] = sb_eboss[ib_eboss[mask_k_sb_2]] + slope_eboss[mask_k_sb_2] * (k_eb[mask_k_sb_2]- deltak_eboss/2 -(klimInf_eboss + ib_eboss[mask_k_sb_2]*deltak_eboss))

        mask_k_sb_3 = (~mask_k_sb_2)&(~mask_k_sb_1)
        slope_eboss[mask_k_sb_3] = (sb_eboss[-1]-sb_eboss[-2])/deltak_eboss
        cor_sb[mask_k_sb_3] = sb_eboss[-1] + slope_eboss[mask_k_sb_3] * (k_eb[mask_k_sb_3]-deltak_eboss/2-(klimSup_eboss-deltak_eboss))    
        
        
        pk_eb = (f_eb[:,2][mask] / (A_lines_eboss[z](k_eb) * A_hcd_eboss[z](k_eb))) - cor_sb
        # pk_eb = (f_eb[:,2][mask] / (A_lines_eboss[z](k_eb) * A_hcd_eboss[z](k_eb)))
        epk_eb = f_eb[:,3][mask] / (A_lines_eboss[z](k_eb) * A_hcd_eboss[z](k_eb))

        # error_bar = np.sqrt(dict_plot[z]["err_to_plot"]**2 + (dict_plot[z]["k_to_plot"] * syste_tot[iz]/np.pi)**2)
        error_bar = dict_plot[z]["err_to_plot"]
        err = ax[0].errorbar(dict_plot[z]["k_to_plot"],
                    dict_plot[z]["p_to_plot"],
                    yerr = error_bar,
                    fmt = ".",
                    color = f"C{iz}",
                    markersize = marker_size)
        

        ax[0].fill_between(k_eb,
                        k_eb * pk_eb/np.pi - k_eb * epk_eb/(np.pi), 
                        k_eb * pk_eb/np.pi + k_eb * epk_eb/(np.pi),
                        color = f"C{iz}",
                        alpha=0.4)   
        

        ax[0].errorbar(k_eb,
                    k_eb * pk_eb/np.pi,
                    k_eb * epk_eb/np.pi,
                    fmt = "None",
                    color = f"C{iz}",
                    markersize = marker_size,
                    alpha=0.2)

        


        pk_desi_interp  = interp1d(dict_plot[z]["k_to_plot"], dict_plot[z]["p_to_plot"], kind='linear',bounds_error=False, fill_value=np.nan)
        epk_desi_interp  = interp1d(dict_plot[z]["k_to_plot"], dict_plot[z]["err_to_plot"], kind='linear',bounds_error=False, fill_value=np.nan)

        ax[1].plot(k_eb,
                    pk_desi_interp(k_eb) / (k_eb * pk_eb/np.pi),
                    marker = marker_style,
                    color = f"C{iz}",
                    markersize = marker_size)   

        ratio_err = (pk_desi_interp(k_eb) / (k_eb * pk_eb/np.pi))*np.sqrt( (epk_desi_interp(k_eb)/pk_desi_interp(k_eb))**2 + (epk_eb/pk_eb)**2)
        mean_error.append(ratio_err)
        
    mean_error = np.nanmean(mean_error,axis=0)

    ax[1].fill_between(k_eb,
                    1 - mean_error, 
                    1 + mean_error, 
                    color = f"k",
                    alpha=0.3,hatch='///')   
        
    legend_elements = [Line2D([], [], color=f"C{i}", marker=None, linestyle='-', label=f'z = {zbins[i]:.1f}') for i in range(len(zbins))]

    err_artist = copy.deepcopy(err)
    err_artist[0].set_color("k")
    err_artist.set_label("This work")
    err_artist[2][0].set_color("k")
    legend_elements = legend_elements + [err_artist,
                                        mpatches.Patch(color='k', alpha=0.4, label='eBOSS DR14')]

    ax[0].legend(handles=legend_elements,loc=2, bbox_to_anchor=(1.03, 0.9), borderaxespad=0.,fontsize = fontlegend)
        

    legend_elements = [Line2D([], [], color=f"k", marker=".", linestyle='-', label=f'Ratio'),
                    mpatches.Patch(color='k', alpha=0.4, label='Mean error',hatch='///')]

    ax[1].legend(handles=legend_elements,loc=2, bbox_to_anchor=(1.03, 0.9), borderaxespad=0.,fontsize = fontlegend)

        
    ax[1].set_xlabel(r'$k~[$' + '$\mathrm{s}$' + r'$\cdot$' + '$\mathrm{km}^{-1}$'  + '$]$', fontsize = fontsize)
    ax[0].set_ylabel(r'$\Delta_{1\mathrm{D},\alpha}^{2}$', fontsize = fontsize)

    ax[0].set_yscale('log')
    ax[1].set_ylabel('DESI/eBOSS', fontsize = fontsize)

    ax[0].set_xlim(0.0, 0.02)
    ax[0].set_ylim(0.008, 0.2)
    ax[1].set_ylim(0.8,1.2)


    ticks = np.arange(0.0, 0.021, 0.005)

    ax[1].set_xticks(ticks)

    ax[0].yaxis.set_tick_params(labelsize=ysize)
    ax[1].xaxis.set_tick_params(labelsize=xsize)
    ax[1].yaxis.set_tick_params(labelsize=ysize)

    fig.tight_layout()

    fig.savefig("p1d_comparison_to_eBOSS.pdf")
    fig.savefig("p1d_comparison_to_eBOSS.png",dpi=200)


def plot_diff_figure(outname,
                     zbins,
                     dict_plot,
                     kmax,
                     colors,
                     reslabel,
                     reslabel2):
    diff_data_model=[]
    chi_data_model=[]
    mask_k = dict_plot[zbins[0]]["diff_k_to_plot"]<kmax
    for iz,z in enumerate(zbins):
        diff_data_model.append(dict_plot[z]["diff_p_to_plot"][mask_k])
        chi_data_model.append(dict_plot[z]["chi_p_to_plot"][mask_k])
    plt.figure()
    sns.violinplot(data=pandas.DataFrame(np.array(diff_data_model).T,None,zbins),
                  inner=None,orient='v',palette=colors,scale='width')
    for i,d in enumerate(diff_data_model):
        plt.errorbar(i,np.mean(d),scipy.stats.sem(d,ddof=0), color='0.3',marker='.')
    plt.xlabel('z')
    plt.ylabel('$(P-P_{model})/P$')
    plt.tight_layout()
    plt.savefig(outname+f"_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}_diff.pdf")
    plt.figure()
    sns.violinplot(data=pandas.DataFrame(np.array(chi_data_model).T,None,zbins),inner=None,orient='v',palette=colors,scale='width')
    for i,d in enumerate(chi_data_model):
        plt.errorbar(i,np.mean(d),scipy.stats.sem(d,ddof=0), color='0.3',marker='.')
    plt.xlabel('z')
    plt.ylabel('$(P-P_{model})/\sigma_P}$')
    plt.tight_layout()
    plt.savefig(outname+f"_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}_chi.pdf")
 """


# Line plots

def plot_lines_study(multiple_data,
                     zbins,
                     out_name,
                     k_units,
                     **kwargs):
    for i in range(len(multiple_data)):
        mean_dict = return_mean_z_dict(zbins,multiple_data[i])
        mean_dict["k_array"],mean_dict["meanPk"]

    return()





# Uncertainties plots

def plot_uncertainties(data,
                       zbins,
                       out_name,
                       k_units,
                       **kwargs):
    return()
