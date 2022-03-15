#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""


import numpy as np
import matplotlib.pyplot as plt
import fitsio
import seaborn as sns
import pandas
from matplotlib.ticker import FuncFormatter
from functools import partial
import scipy
from matplotlib.lines import Line2D
import struct
import astropy.table as t

from p1desi import utils


def read_pk_means(pk_means_name):
    pkmeans = t.Table.read(pk_means_name)
    return(pkmeans)



### P1D plots

def load_model(model,model_file):

    if model == "eBOSSmodel_stack" :
        eBOSSmodel_lowz=read_in_model(model_file[0])
        eBOSSmodel_highz=read_in_model(model_file[1])
        eBOSSmodel_stack=[np.vstack([m,m2]) for m,m2 in zip(eBOSSmodel_lowz, eBOSSmodel_highz)]
        return(eBOSSmodel_stack)
    elif model == "Naimmodel_stack":
        def naim_function4(k,z,k0=0.009,k1=0.053,z0=3,A=0.066,B=3.59,n=-2.685,alpha=-0.22,beta=-0.16):
            knorm0=k/k0
            knorm1=k/k1
            exp1=3+n+alpha*np.log(knorm0)
            exp2=B+beta*np.log(knorm0)
            nom=knorm0**exp1
            denom=1+knorm1**2
            zfac=(1+z)/(1+z0)
            return A*nom/denom*zfac**exp2

        Naimmodel={}
        z_array=np.arange(2.2,4.7,0.2)
        k_array=np.arange(0.001,0.1,0.0001)
        Naimmodel['kpk']=naim_function4(k_array[np.newaxis,:],z_array[:,np.newaxis],A=0.084,B=3.64,alpha=-0.155,beta=0.32,k1=0.048,n=-2.655)
        kk , zz = np.meshgrid(k_array,z_array)
        Naimmodel['k'] = kk
        Naimmodel['z'] = zz

        Naimmodel_stack=(np.array(Naimmodel['z']),np.array(Naimmodel['k']),np.array(Naimmodel['kpk']))
        return(Naimmodel_stack)

    elif model == "Naimmodel_truth_mocks":
        def readTrueP1D(fname):
            file = open(fname, 'rb')
            nk, nz = struct.unpack('ii', file.read(struct.calcsize('ii')))

            fmt = 'd' * nz
            data = file.read(struct.calcsize(fmt))
            z = np.array(struct.unpack(fmt, data), dtype=np.double)

            fmt =  'd' * nk
            data = file.read(struct.calcsize(fmt))
            k = np.array(struct.unpack(fmt, data), dtype=np.double)

            fmt =  'd' * nk * nz
            data = file.read(struct.calcsize(fmt))
            p1d = np.array(struct.unpack(fmt, data), dtype=np.double).reshape((nz, nk))

            return z, k, p1d

        z, k, p = readTrueP1D(model_file)
        Naimmodel={}
        Naimmodel['z']=np.array([[z[i] for j in range(len(k))] for i in range(len(z))])
        Naimmodel['k']=np.array([k for i in range(len(z))])
        Naimmodel['kpk']=p * k / np.pi
        Naimmodel_mock=(np.array(Naimmodel['z']),np.array(Naimmodel['k']),np.array(Naimmodel['kpk']))
        return(Naimmodel_mock)
    else :
        raise ValueError("Incorrect model")





def read_in_model(filename):
    tab=fitsio.FITS(filename)[1]
    z=tab['z'][:].reshape(-1,1000)
    k=tab['k'][:].reshape(-1,1000)
    kpk=tab['kpk'][:].reshape(-1,1000)
    return z,k,kpk



def kAAtokskm(x, pos=None,z=2.2):
    kstr=x
    c=3e5
    lya=1216
    knew=float(kstr)/(c/(1+z)/lya)
    transformed_label='{:.3f}'.format(knew)
    return transformed_label

def kskmtokAA(x,z=2.2):
    kstr=x
    c=3e5
    lya=1216
    knew=float(kstr)*(c/(1+z)/lya)
    transformed_label='{:.3f}'.format(knew)
    return transformed_label

def convert_data_to_kms(data):
    c=3e5
    lya=1216
    scale_fac = (c/(1+data["z"])/lya)
    if(data is not None):
        for key in data.keys():
            if((key!="rescor")|(key!="z")|(key!="nmodes")):
                if(key =="k"):
                    data[key]=np.transpose(np.transpose(data[key])/scale_fac)
                else :
                    data[key]=np.transpose(np.transpose(data[key])*scale_fac)
    return(data)


def adjust_fig(fig,ax,ax2,fontt):
    #this createss more x-axes to compare things in k[s/km]
    fig.subplots_adjust(top=0.75)
    par1 = ax.twiny()
    par2 = ax.twiny()
    par3 = ax.twiny()
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["top"].set_position(("axes", 1.2))
    par3.spines["top"].set_position(("axes", 1.4))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["top"].set_visible(True)
    par1.set_xlabel(r' k [s/km] @ z=2.2', fontsize = fontt)
    par2.set_xlabel(r' k [s/km] @ z=2.8', fontsize = fontt)
    par3.set_xlabel(r' k [s/km] @ z=3.4', fontsize = fontt)

    par1.set_xlim(*ax2.get_xlim())
    par2.set_xlim(*ax2.get_xlim())
    par3.set_xlim(*ax2.get_xlim())

    par1.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm,z=2.2)))
    par2.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm,z=2.8)))
    par3.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm,z=3.4)))

    return(par1,par2,par3)



def place_k_speed_unit_axis(fig,ax,fontt=None):
    #this createss more x-axes to compare things in k[s/km]
    par1 = ax.twiny()
    par2 = ax.twiny()
    par3 = ax.twiny()
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["top"].set_position(("axes", 1.3))
    par3.spines["top"].set_position(("axes", 1.6))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["top"].set_visible(True)
    par1.set_xlabel(r' k [s/km] @ z=2.2', fontsize = fontt)
    par2.set_xlabel(r' k [s/km] @ z=2.8', fontsize = fontt)
    par3.set_xlabel(r' k [s/km] @ z=3.4', fontsize = fontt)

    par1.set_xlim(*ax.get_xlim())
    par2.set_xlim(*ax.get_xlim())
    par3.set_xlim(*ax.get_xlim())

    par1.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm,z=2.2)))
    par2.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm,z=2.8)))
    par3.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm,z=3.4)))


def place_k_wavelength_unit_axis(fig,ax,z,fontt=None):
    #this createss more x-axes to compare things in k[s/km]
    par1 = ax.twiny()
    par1.set_xlabel(r' k [s/km] @ z=2.2', fontsize = fontt)
    par1.xaxis.set_major_formatter(FuncFormatter(partial(kskmtokAA,z)))







def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)




def prepare_plot_values(data,
                        zbins,
                        comparison=None,
                        comparison_model=None,
                        comparison_model_file=None,
                        plot_P=False,
                        z_binsize=0.2,
                        velunits=False,
                        substract_sb=None,
                        substract_sb_comparison=True):

    dict_plot = {}

    if plot_P:
        meanvar='meanPk'
        errvar='errorPk'
    else:
        meanvar='meanDelta2'
        errvar='errorDelta2'

    if comparison_model is not None:
        zmodel,kmodel,kpkmodel = load_model(comparison_model,comparison_model_file)
    minrescor=np.inf
    maxrescor=0.0

    for iz,z in enumerate(zbins):
        dict_plot[z] = {}
        dat=data[iz]
        select=dat['N']>0
        if(substract_sb is not None):
            dat_sb=substract_sb[iz]
            p_sb = dat_sb[meanvar][select]
        k_to_plot=np.array(dat['meank'][select])
        p_to_plot=dat[meanvar][select]
        if(substract_sb is not None):
            p_to_plot = p_to_plot - p_sb
        err_to_plot=dat[errvar][select]


        if (comparison_model is not None) & (comparison is not None):
            raise ValueError("Please choose between plotting a model or another P1D as a comparison")

        if comparison_model is not None:
            izmodel=np.abs((zmodel-z))<z_binsize/2
            izmodel=izmodel.nonzero()[0][0]
            if velunits:
                convfactor=1
            else:
                convfactor=3e5/(1215.67*(1+zmodel[izmodel,0]))
            if plot_P:
                k_to_plot_comparison = kmodel[izmodel,:]*convfactor
                p_to_plot_comparison = (1/convfactor)*kpkmodel[izmodel,:]/kmodel[izmodel,:]*np.pi
            else:
                k_to_plot_comparison = kmodel[izmodel,:]*convfactor
                p_to_plot_comparison = kpkmodel[izmodel,:]
            err_to_plot_comparison = None


        if comparison is not None:
            k_to_plot_comparison = comparison['meank'][iz,:]
            p_to_plot_comparison = comparison[meanvar][iz,:]
            if(substract_sb is not None):
                if(substract_sb_comparison):
                    p_to_plot_comparison = p_to_plot_comparison - p_sb
            err_to_plot_comparison = comparison[errvar][iz,:]

        ## Comparison

        if (comparison_model is None) & (comparison is None):
            k_to_plot_comparison = None
            p_to_plot_comparison = None
            err_to_plot_comparison = None
            diff_k_to_plot = None
            diff_p_to_plot = None
            chi_p_to_plot = None
            diff_err_to_plot = None
        else:

            inter=scipy.interpolate.interp1d(k_to_plot_comparison,p_to_plot_comparison,fill_value='extrapolate')
            p_comparison_interp=inter(k_to_plot)
            diff_k_to_plot = k_to_plot
            diff_p_to_plot = (p_to_plot-p_comparison_interp)/p_comparison_interp
            chi_p_to_plot = (p_to_plot-p_comparison_interp)/err_to_plot
            if(err_to_plot_comparison is None):
                diff_err_to_plot = err_to_plot/p_comparison_interp
            else:
                inter_err=scipy.interpolate.interp1d(k_to_plot_comparison,err_to_plot_comparison,fill_value='extrapolate')
                err_comparison_interp=inter_err(k_to_plot)
                diff_err_to_plot = (p_to_plot/p_comparison_interp)*np.sqrt((err_to_plot/p_to_plot)**2 + (err_comparison_interp/p_comparison_interp)**2)



        if('rescor' in dat.colnames):
            try:
                if np.max(dict_plot[z]["k_to_plot"])>0:
                    minrescor=np.min([minrescor,
                                      np.min(dict_plot[z]["k_to_plot"][(dat['rescor'][select]<0.1)&(dat['rescor'][select]>0)])])
                    maxrescor=np.max([maxrescor,
                                      np.min(dict_plot[z]["k_to_plot"][(dat['rescor'][select]<0.1)&(dat['rescor'][select]>0)])])
            except:
                print('rescor information not computed, skipping')

        dict_plot["minrescor"] = minrescor
        dict_plot["maxrescor"] = maxrescor


        dict_plot[z]["number_chunks"] = dat['N_chunks']
        dict_plot[z]["k_to_plot"] = k_to_plot
        dict_plot[z]["p_to_plot"] = p_to_plot
        dict_plot[z]["err_to_plot"] = err_to_plot
        dict_plot[z]["k_to_plot_comparison"] = k_to_plot_comparison
        dict_plot[z]["p_to_plot_comparison"] = p_to_plot_comparison
        dict_plot[z]["err_to_plot_comparison"] = err_to_plot_comparison
        dict_plot[z]["diff_k_to_plot"] = diff_k_to_plot
        dict_plot[z]["diff_p_to_plot"] = diff_p_to_plot
        dict_plot[z]["chi_p_to_plot"] = chi_p_to_plot
        dict_plot[z]["diff_err_to_plot"] = diff_err_to_plot


    return(dict_plot)



def plot_data(data,
              zbins,
              outname,
              plot_P=False,
              comparison=None,
              comparison_model=None,
              comparison_model_file=None,
              plot_diff=False,
              substract_sb=None,
              **kwargs):

    velunits = data.meta["VELUNITS"]

    res_label = utils.return_key(kwargs,"res_label","")
    res_label2 = utils.return_key(kwargs,"res_label2","")
    diff_range = utils.return_key(kwargs,"diff_range",0.4)
    no_errors_diff = utils.return_key(kwargs,"no_errors_diff",False)
    marker_size = utils.return_key(kwargs,"marker_size",6)
    marker_style = utils.return_key(kwargs,"marker_style","o")
    fonttext = utils.return_key(kwargs,"fonttext",None)
    fontlab = utils.return_key(kwargs,"fontlab",None)
    fontlegend = utils.return_key(kwargs,"fontl",None)
    z_binsize = utils.return_key(kwargs,"z_binsize",0.2)
    colors = utils.return_key(kwargs,"colors",sns.color_palette('deep',len(zbins)))
    kmin = utils.return_key(kwargs,"kmin",4e-2)
    kmax = utils.return_key(kwargs,"kmax",2.5)
    ymin = utils.return_key(kwargs,"ymin",None)
    ymax = utils.return_key(kwargs,"ymax",None)
    grid = utils.return_key(kwargs,"grid",True)
    substract_sb_comparison = utils.return_key(kwargs,"substract_sb_comparison",True)


    if(comparison is not None):
        comparison_data = read_pk_means(comparison)
    else:
        comparison_data = None


    comparison_plot_style = utils.return_key(kwargs,"comparison_plot_style",None)


    fig,(ax,ax2) = plt.subplots(2,figsize = (8, 8),gridspec_kw=dict(height_ratios=[3,1]),sharex=True)
    if not velunits:
        par1,par2,par3 = adjust_fig(fig,ax,ax2,fonttext)


    dict_plot = prepare_plot_values(data,
                                    zbins,
                                    comparison=comparison_data,
                                    comparison_model=comparison_model,
                                    comparison_model_file=comparison_model_file,
                                    plot_P=plot_P,
                                    z_binsize=z_binsize,
                                    velunits=velunits,
                                    substract_sb=substract_sb,
                                    substract_sb_comparison=substract_sb_comparison)


    for iz,z in enumerate(zbins):
        ax.errorbar(dict_plot[z]["k_to_plot"],
                    dict_plot[z]["p_to_plot"],
                    yerr =dict_plot[z]["err_to_plot"],
                    fmt = marker_style,
                    color = colors[iz],
                    markersize = marker_size,
                    label =r' z = {:1.1f}, {} chunks'.format(z,dict_plot[z]["number_chunks"]))

        if(dict_plot[z]["k_to_plot_comparison"] is not None):
            if((comparison_plot_style == "fill")&(dict_plot[z]["err_to_plot_comparison"] is not None)):
                ax.fill_between(dict_plot[z]["k_to_plot_comparison"],
                                dict_plot[z]["p_to_plot_comparison"]-dict_plot[z]["err_to_plot_comparison"],
                                dict_plot[z]["p_to_plot_comparison"]+dict_plot[z]["err_to_plot_comparison"],
                                alpha=0.5,
                                color = colors[iz])
            else:
                if(dict_plot[z]["err_to_plot_comparison"] is not None):
                    ax.errorbar(dict_plot[z]["k_to_plot_comparison"],
                                dict_plot[z]["p_to_plot_comparison"],
                                dict_plot[z]["err_to_plot_comparison"],
                                color = colors[iz],ls=':')
                else:
                    ax.plot(dict_plot[z]["k_to_plot_comparison"],
                            dict_plot[z]["p_to_plot_comparison"],
                            color = colors[iz],ls=':')
            if(no_errors_diff):
                ax2.plot(dict_plot[z]["diff_k_to_plot"],
                         dict_plot[z]["diff_p_to_plot"],
                         color=colors[iz],label='',marker='.',ls='',zorder=-iz)
            else:
                ax2.errorbar(dict_plot[z]["diff_k_to_plot"],
                             dict_plot[z]["diff_p_to_plot"],
                             dict_plot[z]["diff_err_to_plot"],
                             color=colors[iz],label='',ls='',marker='.',zorder=-iz)


    if((dict_plot["minrescor"]!=np.inf)|(dict_plot["minrescor"]!=0.0)):
        ax.fill_betweenx([-1000,1000],
                         [dict_plot["minrescor"],dict_plot["minrescor"]],
                         [dict_plot["maxrescor"],dict_plot["maxrescor"]],
                         color='0.7',zorder=-30)
        ax2.fill_betweenx([-1000,1000],
                          [dict_plot["minrescor"],dict_plot["minrescor"]],
                          [dict_plot["maxrescor"],dict_plot["maxrescor"]],
                          color='0.7',zorder=-30,label='')

    if velunits:
        ax2.set_xlabel(r' k [s/km]', fontsize = fonttext)
    else:
        ax2.set_xlabel(r' k [1/$\AA$]', fontsize = fonttext)

    if plot_P:
        ax.set_ylabel(r'$P_{1d}$ ', fontsize=fonttext, labelpad=-1)
        ax2.set_ylabel(r'$(P_{1d,data}-P_{1d,comp})/P_{1d,comp}$')
    else:
        ax.set_ylabel(r'$\Delta^2_{1d}$ ', fontsize=fonttext, labelpad=-1)
        ax2.set_ylabel(r'$(\Delta^2_{1d,data}-\Delta^2_{1d,comp})/\Delta^2_{1d,comp}$')

    ax.set_yscale('log')

    for a in ax,ax2:
        if(grid):
            a.grid()
        a.xaxis.set_ticks_position('both')
        a.xaxis.set_tick_params(direction='in')
        a.yaxis.set_ticks_position('both')
        a.yaxis.set_tick_params(direction='in')
        a.xaxis.set_tick_params(labelsize=fontlab)
        a.yaxis.set_tick_params(labelsize=fontlab)
        a.set_xlim(kmin,kmax)
    if(ymin is None):
        if not plot_P:
            ax.set_ylim(4e-3,2)
        else:
            if not velunits:
                ax.set_ylim(0.01,0.5)
            else:
                ax.set_ylim(1,300)
    else:
        ax.set_ylim(ymin,ymax)

    ax2.set_ylim(-diff_range/2,diff_range/2)
    handles, labels = ax.get_legend_handles_labels()

    legend1 = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.03, -0.5, 0.4, 1.0), borderaxespad=0.,fontsize = fontlegend)

    ax.errorbar([0],[0], yerr =[0], fmt = marker_style,color='k', markersize = marker_size, label ='{}'.format(res_label))
    if (comparison_plot_style == "fill"):
        ax.fill_between([0],[0],[0],label=res_label2,color='k')
    else:
        ax.plot([0],[0],label=res_label2,color='k',ls=':')

    handles, labels = ax.get_legend_handles_labels()
    handles,labels=zip(*[(h,l) for (h,l) in zip(handles,labels) if not 'z =' in l])
    ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.03, 0.9), borderaxespad=0.,fontsize = fontlegend)

    if not velunits:
        par1.set_xlim(*ax2.get_xlim())
        par2.set_xlim(*ax2.get_xlim())
        par3.set_xlim(*ax2.get_xlim())

    ax.add_artist(legend1)
    fig.subplots_adjust(top=0.75,bottom=0.1,left=0.1,right=0.65,hspace=0.2,wspace=0.2)
    fig.savefig(outname+f"{'' if not plot_P else '_powernotDelta'}_kmax_{kmax}.pdf")


    if plot_diff:
        plot_diff_figure(outname,
                         zbins,
                         dict_plot,
                         kmax,
                         colors,
                         res_label,
                         res_label2)



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



#
# def plot_differences(name_ref,
#                      name_comparison,
#                      zbins,
#                      outname,
#                      plot_P=False,
#                      comparison=None,
#                      comparison_model=None,
#                      comparison_model_file=None,
#                      plot_diff=False,
#                      substract_sb=None,
#                      **kwargs):
#
#
#     data_ref = read_pk_means(name_ref)
#
#     if comparison is not None:
#         k_to_plot_comparison = comparison['meank'][iz,:]
#         p_to_plot_comparison = comparison[meanvar][iz,:]
#         if(substract_sb is not None):
#             p_to_plot_comparison = p_to_plot_comparison - p_sb
#         err_to_plot_comparison = comparison[errvar][iz,:]
#
#
#     # velunits = data.meta["VELUNITS"]
#     #
#     # res_label = utils.return_key(kwargs,"res_label","")
#     # res_label2 = utils.return_key(kwargs,"res_label2","")
#     # diff_range = utils.return_key(kwargs,"diff_range",0.4)
#     # no_errors_diff = utils.return_key(kwargs,"no_errors_diff",False)
#     # marker_size = utils.return_key(kwargs,"marker_size",6)
#     # marker_style = utils.return_key(kwargs,"marker_style","o")
#     # fonttext = utils.return_key(kwargs,"fonttext",None)
#     # fontlab = utils.return_key(kwargs,"fontlab",None)
#     # fontlegend = utils.return_key(kwargs,"fontl",None)
#     # z_binsize = utils.return_key(kwargs,"mark_size",0.2)
#     # colors = utils.return_key(kwargs,"colors",sns.color_palette('deep',len(zbins)))
#     # kmin = utils.return_key(kwargs,"kmin",4e-2)
#     # kmax = utils.return_key(kwargs,"kmax",2.5)
#     # grid = utils.return_key(kwargs,"grid",True)
#
#
#
#     comparison_plot_style = utils.return_key(kwargs,"comparison_plot_style",None)
#
#
#     fig,(ax,ax2) = plt.subplots(2,figsize = (8, 8),gridspec_kw=dict(height_ratios=[3,1]),sharex=True)
#     if not velunits:
#         par1,par2,par3 = adjust_fig(fig,ax,ax2,fonttext)
#
#
#     dict_plot = prepare_plot_values(data,
#                                     zbins,
#                                     comparison=comparison_data,
#                                     comparison_model=comparison_model,
#                                     comparison_model_file=comparison_model_file,
#                                     plot_P=plot_P,
#                                     z_binsize=z_binsize,
#                                     velunits=velunits,
#                                     substract_sb=substract_sb)
#
#
#     for iz,z in enumerate(zbins):
#         ax.errorbar(dict_plot[z]["k_to_plot"],
#                     dict_plot[z]["p_to_plot"],
#                     yerr =dict_plot[z]["err_to_plot"],
#                     fmt = marker_style,
#                     color = colors[iz],
#                     markersize = marker_size,
#                     label =r' z = {:1.1f}, {} chunks'.format(z,dict_plot[z]["number_chunks"]))
#
#         if(dict_plot[z]["k_to_plot_comparison"] is not None):
#             if((comparison_plot_style == "fill")&(dict_plot[z]["err_to_plot_comparison"] is not None)):
#                 ax.fill_between(dict_plot[z]["k_to_plot_comparison"],
#                                 dict_plot[z]["p_to_plot_comparison"]-dict_plot[z]["err_to_plot_comparison"],
#                                 dict_plot[z]["p_to_plot_comparison"]+dict_plot[z]["err_to_plot_comparison"],
#                                 alpha=0.5,
#                                 color = colors[iz])
#             else:
#                 if(dict_plot[z]["err_to_plot_comparison"] is not None):
#                     ax.errorbar(dict_plot[z]["k_to_plot_comparison"],
#                                 dict_plot[z]["p_to_plot_comparison"],
#                                 dict_plot[z]["err_to_plot_comparison"],
#                                 color = colors[iz],ls=':')
#                 else:
#                     ax.plot(dict_plot[z]["k_to_plot_comparison"],
#                             dict_plot[z]["p_to_plot_comparison"],
#                             color = colors[iz],ls=':')
#             if(no_errors_diff):
#                 ax2.plot(dict_plot[z]["diff_k_to_plot"],
#                          dict_plot[z]["diff_p_to_plot"],
#                          color=colors[iz],label='',marker='.',ls='',zorder=-iz)
#             else:
#                 ax2.errorbar(dict_plot[z]["diff_k_to_plot"],
#                              dict_plot[z]["diff_p_to_plot"],
#                              dict_plot[z]["diff_err_to_plot"],
#                              color=colors[iz],label='',ls='',marker='.',zorder=-iz)
#
#
#     if((dict_plot["minrescor"]!=np.inf)|(dict_plot["minrescor"]!=0.0)):
#         ax.fill_betweenx([-1000,1000],
#                          [dict_plot["minrescor"],dict_plot["minrescor"]],
#                          [dict_plot["maxrescor"],dict_plot["maxrescor"]],
#                          color='0.7',zorder=-30)
#         ax2.fill_betweenx([-1000,1000],
#                           [dict_plot["minrescor"],dict_plot["minrescor"]],
#                           [dict_plot["maxrescor"],dict_plot["maxrescor"]],
#                           color='0.7',zorder=-30,label='')
#
#     if velunits:
#         ax2.set_xlabel(r' k [s/km]', fontsize = fonttext)
#     else:
#         ax2.set_xlabel(r' k [1/$\AA$]', fontsize = fonttext)
#
#     if plot_P:
#         ax.set_ylabel(r'$P_{1d}$ ', fontsize=fonttext, labelpad=-1)
#         ax2.set_ylabel(r'$(P_{1d,data}-P_{1d,comp})/P_{1d,comp}$')
#     else:
#         ax.set_ylabel(r'$\Delta^2_{1d}$ ', fontsize=fonttext, labelpad=-1)
#         ax2.set_ylabel(r'$(\Delta^2_{1d,data}-\Delta^2_{1d,comp})/\Delta^2_{1d,comp}$')
#
#     ax.set_yscale('log')
#
#     for a in ax,ax2:
#         if(grid):
#             a.grid()
#         a.xaxis.set_ticks_position('both')
#         a.xaxis.set_tick_params(direction='in')
#         a.yaxis.set_ticks_position('both')
#         a.yaxis.set_tick_params(direction='in')
#         a.xaxis.set_tick_params(labelsize=fontlab)
#         a.yaxis.set_tick_params(labelsize=fontlab)
#         a.set_xlim(kmin,kmax)
#
#     if not plot_P:
#         ax.set_ylim(4e-3,2)
#     else:
#         if not velunits:
#             ax.set_ylim(0.01,0.5)
#         else:
#             ax.set_ylim(1,300)
#     ax2.set_ylim(-diff_range/2,diff_range/2)
#     handles, labels = ax.get_legend_handles_labels()
#
#     legend1 = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.03, 0.98), borderaxespad=0.,fontsize = fontlegend)
#
#     ax.errorbar([0],[0], yerr =[0], fmt = marker_style,color='k', markersize = marker_size, label ='{}'.format(res_label))
#     if (comparison_plot_style == "fill"):
#         ax.fill_between([0],[0],[0],label=res_label2,color='k')
#     else:
#         ax.plot([0],[0],label=res_label2,color='k',ls=':')
#
#     handles, labels = ax.get_legend_handles_labels()
#     handles,labels=zip(*[(h,l) for (h,l) in zip(handles,labels) if not 'z =' in l])
#     ax2.legend(handles, labels, loc=3, bbox_to_anchor=(1.03, 0.02), borderaxespad=0.,fontsize = fontlegend)
#
#     if not velunits:
#         par1.set_xlim(*ax2.get_xlim())
#         par2.set_xlim(*ax2.get_xlim())
#         par3.set_xlim(*ax2.get_xlim())
#
#     ax.add_artist(legend1)
#     fig.subplots_adjust(top=0.95,bottom=0.114,left=0.078,right=0.758,hspace=0.2,wspace=0.2)
#     fig.tight_layout()
#     fig.savefig(outname+f"{'' if not plot_P else '_powernotDelta'}_kmax_{kmax}.pdf")
#
#
#     if plot_diff:
#         plot_diff_figure(outname,
#                          zbins,
#                          dict_plot,
#                          kmax,
#                          colors,
#                          res_label,
#                          res_label2)
#


### Noise comparison plots


def compute_and_plot_mean_z_noise_power(data,
                                        zbins,
                                        outname,
                                        **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)

    dict_noise_diff=compute_mean_z_noise_power(data,zbins,kmin=kmin,kmax=kmax)
    plot_mean_z_noise_power(dict_noise_diff,zbins,outname)



def compute_mean_z_noise_power(data,zbins,kmin=4e-2,kmax=2.5):
    velunits = data.meta["VELUNITS"]

    if velunits and kmax==2:
        kmax=0.035
    if velunits and kmin==4e-2:
        kmin=8e-4

    diff_model = {"Pk_diff" : [] ,"Pk_noise" : []}
    for iz,z in enumerate(zbins):
        dat=data[iz]
        select=dat['N']>0
        k=dat['meank'][select]
        Pk_noise = dat["meanPk_noise"][select][k<kmax]
        Pk_diff = dat["meanPk_diff"][select][k<kmax]
        diff_model["Pk_diff"].append(Pk_diff)
        diff_model["Pk_noise"].append(Pk_noise)

    dict_noise_diff = {"diff":[],"error_diff" : [],"pipeline" : [],"error_pipeline":[],"diff_over_pipeline":[],"error_diff_over_pipeline":[]}
    for i in range(len(diff_model["Pk_noise"])):
        noise_error = scipy.stats.sem(diff_model["Pk_noise"][i],ddof=0)
        diff_error = scipy.stats.sem(diff_model["Pk_diff"][i],ddof=0)
        diff_over_noise_error = (np.mean(diff_model["Pk_diff"][i])/np.mean(diff_model["Pk_noise"][i]))*np.sqrt((diff_error/np.mean(diff_model["Pk_diff"][i]))**2 + (noise_error/np.mean(diff_model["Pk_noise"][i]))**2)
        dict_noise_diff["diff"].append(np.mean(diff_model["Pk_diff"][i]))
        dict_noise_diff["error_diff"].append(diff_error)
        dict_noise_diff["pipeline"].append(np.mean(diff_model["Pk_noise"][i]))
        dict_noise_diff["error_pipeline"].append(noise_error)
        dict_noise_diff["diff_over_pipeline"].append(np.mean((diff_model["Pk_diff"][i] - diff_model["Pk_noise"][i])/diff_model["Pk_noise"][i]))
        dict_noise_diff["error_diff_over_pipeline"].append(diff_over_noise_error)
    dict_noise_diff["zbins"] = zbins

    return(dict_noise_diff)




def plot_mean_z_noise_power(dict_noise_diff,zbins,outname,dreshift = 0.02):
    fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
    ax[1].set_xticks(zbins)
    for i in range(len(dict_noise_diff["pipeline"])):
        scale_fac = 3e5/((1+zbins[i])*1216)
        pipeline = dict_noise_diff["pipeline"][i] * scale_fac
        error_pipeline = dict_noise_diff["error_pipeline"][i] * scale_fac
        diff = dict_noise_diff["diff"][i] * scale_fac
        error_diff = dict_noise_diff["error_diff"][i] * scale_fac
        diff_over_pipeline = ( dict_noise_diff["diff"][i] - dict_noise_diff["pipeline"][i] ) * scale_fac
        error_diff_over_pipeline = diff_over_pipeline * np.sqrt((error_pipeline/pipeline)**2 + (error_diff/diff)**2)
        ax[0].errorbar(zbins[i],pipeline,error_pipeline,marker='*',color =f"C{i}")
        ax[0].errorbar(zbins[i] + dreshift,diff,error_diff,marker='o',color =f"C{i}")
        ax[1].errorbar(zbins[i],diff_over_pipeline,error_diff_over_pipeline,marker='.')
    legend_elements = [Line2D([], [], color='k', marker='*', linestyle='None', label='$P_{pipeline}$'),
                       Line2D([], [], color='k', marker='o', linestyle='None', label='$P_{diff}$')]
    ax[0].legend(handles=legend_elements)
    ax[0].set_ylabel('$<P> [km/s]$')
    ax[1].set_ylabel('$\delta<P> [km/s]$')
    ax[1].set_xlabel('z')
    ax[1].legend(handles = [Line2D([], [], color='k', marker='None', linestyle='None',
                                  label='Average for all redshift = ${}$%'.format(np.round(np.mean(dict_noise_diff["diff_over_pipeline"])*100,2)))],frameon=False)
    plt.tight_layout()
    fig.savefig("{}_mean_ratio_diff_pipeline_power.pdf".format(outname),format="pdf")



def plot_several_mean_z_noise_power(list_dict,nameout,legend,colors,dreshift = 0.01,marker = [".","^","*","+","x"],obs_wavelength=True):
    marker = marker[0:len(list_dict)]
    fig,ax=plt.subplots(3,1,figsize=(8,8),sharex=True)
    displacement = np.array([(i - len(list_dict)//2 - 0.5*(len(list_dict)%2 -1))*dreshift for i in range(len(list_dict))])
    if(obs_wavelength):
        lya = 1215.673123130217
        zbins = np.round((1+list_dict[0]["zbins"])*lya,0)
        displacement = displacement*lya
        ax[2].set_xlabel('Observed wavelength [$\AA$]')
    else :
        zbins = list_dict[0]["zbins"]
        ax[2].set_xlabel('z')
    ax[2].set_xticks(zbins)
    for i in range(len(list_dict)):
        for j in range(len(list_dict[i]["diff"])):
            ax[0].errorbar(zbins[j] + displacement[i],list_dict[i]["pipeline"][j],list_dict[i]["error_pipeline"][j],marker=marker[i],ecolor=colors[j],color=colors[j])
            ax[1].errorbar(zbins[j] + displacement[i],list_dict[i]["diff"][j],list_dict[i]["error_diff"][j],marker=marker[i],ecolor=colors[j],color=colors[j])
            ax[2].errorbar(zbins[j] + displacement[i],list_dict[i]["diff_over_pipeline"][j],list_dict[i]["error_diff_over_pipeline"][j],marker=marker[i],ecolor=colors[j],color=colors[j])
    legend_elements = [Line2D([0], [0], marker=marker[i], color='k', label=legend[i]) for i in range(len(marker))]
    ax[0].legend(handles=legend_elements)
    ax[0].set_ylabel('$mean_{k}(P_{pipeline}) [\AA]$')
    ax[1].set_ylabel('$mean_{k}(P_{diff}) [\AA]$')
    ax[2].set_ylabel('$mean_{k}((P_{diff}-P_{pipeline})/P_{pipeline})$')
    plt.tight_layout()
    fig.savefig("{}_mean_ratio_diff_pipeline_power.pdf".format(nameout),format="pdf")



# Noise power ratio


def plot_noise_comparison_function(zbins,
                                   data,
                                   out_name,
                                   mean_dict,
                                   k_units,
                                   **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)
    fig2,ax2=plt.subplots(4,1,figsize=(8,10),sharex=True)
    for z,d in zip(zbins,data):
        ax2[0].plot(d['meank'],d["meanPk_noise"],label=f'{z:.1f}')
        if(k_units == "A"):
            ax2[0].set_ylabel("$P_{pipeline} [\AA]$")
        elif(k_units == "kms"):
            ax2[0].set_ylabel("$P_{pipeline} [km/s]$")
        ax2[1].plot(d['meank'],d["meanPk_diff"],label=f'{z:.1f}')
        if(k_units == "A"):
            ax2[1].set_ylabel("$P_{diff} [\AA]$")
        elif(k_units == "kms"):
            ax2[1].set_ylabel("$P_{diff} [km/s]$")
        yerr = (d['errorPk_diff'] / d['meanPk_noise']) *  np.sqrt((d['errorPk_noise'] / d['meanPk_noise'])**2 + (d['errorPk_diff'] / d['meanPk_diff'])**2)
        ax2[2].errorbar(d['meank'],(d['meanPk_diff'] -d['meanPk_noise']) / d['meanPk_noise'], yerr = yerr , fmt = 'o')#,marker_size=6)
        ax2[2].set_ylabel('$(P_{diff} - P_{pipeline})/P_{pipeline}$')
        ax2[0].legend()
    ax2[3].errorbar(mean_dict["k_array"],(mean_dict["meanPk_diff"] - mean_dict["meanPk_noise"]) / mean_dict["meanPk_noise"], yerr =mean_dict["error_diffovernoise"], fmt = 'o')#,marker_size=6)
    ax2[3].set_ylabel('$mean_{z}((P_{diff} - P_{pipeline})/P_{pipeline})$')
    if(k_units == "A"):
        ax2[3].set_xlabel('k[1/$\AA$]')
        place_k_speed_unit_axis(fig2,ax2[0])
    elif(k_units == "kms"):
        ax2[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax2[0].set_xlim(kmin,kmax)
    fig2.tight_layout()
    fig2.savefig(f"{out_name}_ratio_diff_pipeline_power_unit{k_units}.pdf",format="pdf")


def plot_side_band_study(zbins,
                         data,
                         out_name,
                         mean_dict,
                         noise_to_plot,
                         labelnoise,
                         k_units,
                         side_band_legend,
                         side_band_comp = None,
                         **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)
    fig3,ax3=plt.subplots(4,1,figsize=(8,10),sharex=True)

    for z,d in zip(zbins,data):
        ax3[0].plot(d['meank'],d['meanPk_raw'],label=f'{z:.1f}')
        if(k_units == "A"):
            ax3[0].set_ylabel('$P_{raw} [\AA]$')
        elif(k_units == "kms"):
            ax3[0].set_ylabel('$P_{raw} [km/s]$')
        ax3[0].legend()
        ax3[1].plot(d['meank'],d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax3[1].set_ylabel('$P_{' + labelnoise +'} [\AA]$')
        elif(k_units == "kms"):
            ax3[1].set_ylabel('$P_{' + labelnoise +'} [km/s]$')
        ax3[2].plot(d['meank'],d['meanPk_raw'] - d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax3[2].set_ylabel('$ (P_{raw} - P_{pipeline}) [\AA]$')
        elif(k_units == "kms"):
            ax3[2].set_ylabel('$ (P_{raw} - P_{pipeline}) [km/s]$')

    ax3[3].errorbar(mean_dict["k_array"],mean_dict["meanPk"],mean_dict["errorPk"], fmt = 'o',label=side_band_legend[0])
    if(k_units == "A"):
        ax3[3].set_ylabel('$mean_{z}(P_{SB}) [\AA]$')
    elif(k_units == "kms"):
        ax3[3].set_ylabel('$mean_{z}(P_{SB}) [km/s]$')
    poly = scipy.polyfit(mean_dict["k_array"],mean_dict["meanPk"],6)
    Poly = np.polynomial.polynomial.Polynomial(np.flip(poly))
    cont_k_array = np.linspace(np.min(mean_dict["k_array"]),np.max(mean_dict["k_array"]),300)
    polynome = Poly(cont_k_array)
    mean_dict["poly"]= polynome
    mean_dict["k_cont"]=cont_k_array
    ax3[3].plot(cont_k_array,polynome)
    if(side_band_comp is not None):
        yerr =np.sqrt( side_band_comp["error_meanPk_noise"]**2 + side_band_comp["error_meanPk_raw"]**2)
        ax3[3].errorbar(side_band_comp["k_array"],side_band_comp["meanPk_raw"] - side_band_comp["meanPk_noise"],yerr, fmt = 'o',label=side_band_legend[1])
        ax3[3].plot(side_band_comp["k_cont"],side_band_comp["poly"])
        ax3[3].legend()
    if(k_units == "A"):
        ax3[3].set_xlabel('k[1/$\AA$]')
        place_k_speed_unit_axis(fig3,ax3[0])
    elif(k_units == "kms"):
        ax3[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax3[0].set_xlim(kmin,kmax)
    fig3.tight_layout()
    fig3.savefig(f"{out_name}_side_band_unit{k_units}.pdf",format="pdf")



def plot_noise_power_ratio(data,
                           zbins,
                           out_name,
                           mean_dict,
                           noise_to_plot,
                           labelnoise,
                           k_units,
                           fit_asymptote= False,
                           **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)
    ncol_legend = utils.return_key(kwargs,"ncol_legend",2)
    fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)
    for z,d in zip(zbins,data):
        ax[0].plot(d['meank'],d['meanPk_raw'],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[0].set_ylabel('$P_{raw} [\AA]$')
        elif(k_units == "kms"):
            ax[0].set_ylabel('$P_{raw} [km/s]$')
        ax[0].legend(ncol=ncol_legend)
        ax[1].plot(d['meank'],d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [\AA]$')
        elif(k_units == "kms"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [km/s]$')
        ax[2].plot(d['meank'],d[noise_to_plot]/d['meanPk_raw'],label=f'{z:.1f}')
        ax[2].set_ylabel('$P_{' + labelnoise +'}/P_{raw}$')


    ax[3].errorbar(mean_dict["k_array"],mean_dict[noise_to_plot]/mean_dict["meanPk_raw"], yerr =mean_dict["error_{}overraw".format(noise_to_plot)], fmt = 'o')#,marker_size=6)
    ax[3].set_ylabel('$mean_{z}(P_{' + labelnoise +'}/P_{raw})$')
    if(fit_asymptote):
        try :
            f_const = lambda x,a : np.array([a for i in range(len(x))])
            if(np.min(mean_dict["k_array"])> 3.0) : kmin_fit = np.min(mean_dict["k_array"])
            else : kmin_fit = 3.0
            mask = mean_dict["k_array"] > kmin_fit
            arg_func = scipy.optimize.curve_fit(f_const,
                                                mean_dict["k_array"][mask],
                                                (mean_dict[noise_to_plot]/mean_dict["meanPk_raw"])[mask],
                                                p0=[1])
            cont_k_array = np.linspace(kmin_fit,np.max(mean_dict["k_array"]),500)
            fit_exp = f_const(cont_k_array,*arg_func[0])
            fit_exp_min = f_const(cont_k_array,*(arg_func[0]-np.diag(arg_func[1])))
            fit_exp_max = f_const(cont_k_array,*(arg_func[0]+np.diag(arg_func[1])))
            ax[3].fill_between(cont_k_array, fit_exp_min, fit_exp_max,facecolor='grey', interpolate=True,alpha=0.5)
            ax[3].plot(cont_k_array,fit_exp)
            ax[3].legend(["asymptote = {}".format(np.round(arg_func[0][0],3))])
        except:
            print("Pdiff over Praw fit did not converge")
    if(k_units == "A"):
        ax[3].set_xlabel('k[1/$\AA$]')
        place_k_speed_unit_axis(fig,ax[0])
    elif(k_units == "kms"):
        ax[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax[0].set_xlim(kmin,kmax)
    for i in [0,1,2,3]:
        ax[i].grid()
    ax[2].set_ylim(0.0,1.1)
    ax[3].set_ylim(0.0,1.1)
    fig.tight_layout()
    fig.savefig(f"{out_name}_ratio_{labelnoise}_raw_power_unit{k_units}.pdf",format="pdf")


def return_mean_z_dict(zbins,data):
    mean_dict = {"meanPk_diff" : [],"meanPk_noise" : [],"meanPk_raw" : [], "error_diffovernoise":[], "error_meanPk_diffoverraw":[], "error_meanPk_noiseoverraw":[],"error_meanPk_raw":[],"k_array":[],"error_meanPk_noise" : [],"errorPk" : [],"meanPk" : []}
    for z,d in zip(zbins,data):
        mean_dict["meanPk"].append(d['meanPk'])
        mean_dict["errorPk"].append(d['errorPk'])
        mean_dict["meanPk_noise"].append(d['meanPk_noise'])
        mean_dict["meanPk_raw"].append(d['meanPk_raw'])
        yerr = (d['errorPk_noise'] / d['meanPk_raw']) *  np.sqrt((d['errorPk_raw'] / d['meanPk_raw'])**2 + (d['errorPk_noise'] / d['meanPk_noise'])**2)
        mean_dict["error_meanPk_noiseoverraw"].append(yerr)
        mean_dict["k_array"] = d['meank']
        mean_dict["error_meanPk_raw"].append(d['errorPk_raw'])
        mean_dict["error_meanPk_noise"].append(d['errorPk_noise'])

        mean_dict["meanPk_diff"].append(d['meanPk_diff'])
        mean_dict["error_diffovernoise"].append(yerr)
        yerr = (d['errorPk_diff'] / d['meanPk_raw']) *  np.sqrt((d['errorPk_raw'] / d['meanPk_raw'])**2 + (d['errorPk_diff'] / d['meanPk_diff'])**2)
        mean_dict["error_meanPk_diffoverraw"].append(yerr)

    mean_dict["meanPk_noise"] = np.mean(mean_dict["meanPk_noise"],axis=0)
    mean_dict["error_meanPk_noiseoverraw"] =np.mean(mean_dict["error_meanPk_noiseoverraw"],axis=0)/np.sqrt(len(mean_dict["error_meanPk_noiseoverraw"]))
    mean_dict["meanPk_raw"] = np.mean(mean_dict["meanPk_raw"],axis=0)
    mean_dict["error_meanPk_raw"] = np.mean(mean_dict['error_meanPk_raw'],axis=0)/np.sqrt(len(mean_dict["error_meanPk_raw"]))
    mean_dict["meanPk"] = np.mean(mean_dict["meanPk"],axis=0)
    mean_dict["errorPk"] = np.mean(mean_dict['errorPk'],axis=0)/np.sqrt(len(mean_dict["errorPk"]))


    mean_dict["error_meanPk_noise"] = np.mean(mean_dict['error_meanPk_noise'],axis=0)/np.sqrt(len(mean_dict["error_meanPk_noise"]))
    mean_dict["meanPk_diff"] = np.mean(mean_dict["meanPk_diff"],axis=0)
    mean_dict["error_meanPk_diffoverraw"] = np.mean(mean_dict["error_meanPk_diffoverraw"],axis=0)/np.sqrt(len(mean_dict["error_meanPk_diffoverraw"]))

    mean_dict["error_diffovernoise"] = np.mean(mean_dict["error_diffovernoise"],axis=0)/np.sqrt(len(mean_dict["error_diffovernoise"]))

    return(mean_dict)



def plot_noise_study(data,
                     zbins,
                     out_name,
                     k_units,
                     use_diff_noise,
                     plot_noise_ratio,
                     plot_noise_comparison,
                     plot_side_band,
                     side_band_comp=None,
                     side_band_legend=["SB1","SB2"],
                     fit_asymptote_ratio= False,
                     **kwargs):


    if(k_units == "A"):
        True
    elif(k_units == "kms"):
        data = convert_data_to_kms(data)
    else:
        raise ValueError("choose units between A and kms")

    mean_dict = return_mean_z_dict(zbins,data)

    if(use_diff_noise):
        noise_to_plot,labelnoise = 'meanPk_diff','diff'
    else:
        noise_to_plot,labelnoise = 'meanPk_noise','pipeline'
    if(plot_noise_ratio):
        plot_noise_power_ratio(data,
                               zbins,
                               out_name,
                               mean_dict,
                               noise_to_plot,
                               labelnoise,
                               k_units,
                               fit_asymptote= fit_asymptote_ratio,
                               **kwargs)
    if(plot_noise_comparison):
        plot_noise_comparison_function(zbins,
                                       data,
                                       out_name,
                                       mean_dict,
                                       k_units,
                                       **kwargs)

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
