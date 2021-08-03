#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""

#note that widget mode needs ipympl installed in the current python environment
#%pylab widget
#%load_ext line_profiler
#%load_ext memory_profiler

import os
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

from p1d_desi import utils


def read_pk_means(pk_means_name):
    pkmeans = t.Table.read(pk_means_name)
    return(pkmeans)



def load_model(name_model,model_dir,model_file=None):

    # model_dir = "/local/home/cravoux/Documents/Python/Data/p1d_models"
    model_dir = "/global/homes/r/ravouxco/1_Documents/Pk1d/models"

    if name_model == "eBOSSmodel_stack" :
        eBOSSmodel_lowz=read_in_model(os.path.join(model_dir,'models_eBOSS_lowz.fits'))
        eBOSSmodel_highz=read_in_model(os.path.join(model_dir,'models_eBOSS_highz.fits'))
        eBOSSmodel_stack=[np.vstack([m,m2]) for m,m2 in zip(eBOSSmodel_lowz, eBOSSmodel_highz)]
        return(eBOSSmodel_stack)
    elif name_model == "DR9model_stack" :
        DR9model_lowz=read_in_model(os.path.join(model_dir,'models_DR9_lowz.fits'))
        DR9model_highz=read_in_model(os.path.join(model_dir,'models_DR9_highz.fits'))
        DR9model_stack=[np.vstack([m,m2]) for m,m2 in zip(DR9model_lowz, DR9model_highz)]
        return(DR9model_stack)
    elif name_model == "Naimmodel_stack":
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
        Naimmodel['z']=np.arange(2.2,4.7,0.2)
        Naimmodel['k']=np.arange(0.001,0.1,0.001)[np.newaxis,:]
        Naimmodel['kpk']=naim_function4(Naimmodel['k'],Naimmodel['z'][:,np.newaxis],A=0.084,B=3.64,alpha=-0.155,beta=0.32,k1=0.048,n=-2.655)
        Naimmodel_stack=(np.array(Naimmodel['z'][:,np.newaxis]),np.array(Naimmodel['k']),np.array(Naimmodel['kpk']))
        return(Naimmodel_stack)

    elif name_model == "Naimmodel_truth_mocks":
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




def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)



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




def prepare_plot_values():
    minrescor=np.inf
    maxrescor=0
    diff_data_model=[]
    chi_data_model=[]
    for iz,z in enumerate(zbins):
        if comparemodel is not None:
            izmodel=np.abs((zmodel-z))<z_binsize/2
            izmodel=izmodel.nonzero()[0][0]

        if not plot_P:
            meanvar='meanDelta2'
            errvar='errorDelta2'
            if comparemodel is not None:
                if velunits:
                    convfactor=1
                    k_to_plot = kmodel[izmodel,:]
                    p_to_plot = kpkmodel[izmodel,:]
                    ax.plot(k_to_plot,p_to_plot, color = colors[iz],ls=':')
                else:
                    convfactor=3e5/(1215.67*(1+zmodel[izmodel,0]))
                    ax.plot(kmodel[izmodel,:]*convfactor,kpkmodel[izmodel,:], color = colors[iz],ls=':')
        else:
            meanvar='meanPk'
            errvar='errorPk'
            if comparemodel is not None:
                if velunits:
                    convfactor=1
                    ax.plot(kmodel[izmodel,:],kpkmodel[izmodel,:]/kmodel[izmodel,:]*np.pi, color = colors[iz],ls=':')
                else:
                    convfactor=3e5/(1215.67*(1+zmodel[izmodel,0]))
                    ax.plot(kmodel[izmodel,:]*convfactor,1/convfactor*kpkmodel[izmodel,:]/kmodel[izmodel,:]*np.pi, color = colors[iz],ls=':')
        if data is not None:
            dat=data[iz]
            select=dat['N']>0
            k=dat['meank'][select]
            P=dat[meanvar][select]
            err=dat[errvar][select]
            ax.errorbar(k,P, yerr =err, fmt = 'o', color = colors[iz], markersize = mark_size, label =r' z = {:1.1f}'.format(z))

        if comparison is not None:
            ax.fill_between(comparison['k'][iz,:],comparison[meanvar][iz,:]-comparison[errvar][iz,:],comparison[meanvar][iz,:]+comparison[errvar][iz,:], alpha=0.5, color = colors[iz], label = (r' z = {:1.1f}'.format(z) if data is None else None))
            if data is not None:
                inter=scipy.interpolate.interp1d(comparison['k'][iz,:],comparison[meanvar][iz,:],fill_value='extrapolate')
            truthint=inter(k)
            if not noerrors:
                ax2.errorbar(k,(P-truthint)/truthint,err/truthint,color=colors[iz],label='',ls='',marker='.',zorder=-iz)
            else:
                ax2.plot(k,(P-truthint)/truthint,color=colors[iz],label='',marker='.',ls='',zorder=-iz)
            diff_data_model.append(((P-truthint)/truthint)[k<kmax])
            chi_data_model.append(((P-truthint)/err)[k<kmax])

        try:
            if np.max(k)>0:
                minrescor=np.min([minrescor,np.min(k[(dat['rescor'][select]<0.1)&(dat['rescor'][select]>0)])])
                maxrescor=np.max([maxrescor,np.min(k[(dat['rescor'][select]<0.1)&(dat['rescor'][select]>0)])])
        except:
            print('rescor information not computed skipping')




def plot_data(data,
              zbins,
              colors,
              outname,
              plot_P,
              comparison=None,
              comparison_model=None,
              comparison_model_file=None,
              kmin=4e-2,
              kmax=2.5,
              z_binsize = 0.2,
              velunits=False,
              **kwargs):


    reslabel = utils.return_key(kwargs,"reslabel","")
    reslabel2 = utils.return_key(kwargs,"reslabel2","")
    diffrange = utils.return_key(kwargs,"diffrange",0.4)
    noerrors = utils.return_key(kwargs,"noerrors",False)
    mark_size = utils.return_key(kwargs,"mark_size",6)
    fontt = utils.return_key(kwargs,"fontt",None)
    fontlab = utils.return_key(kwargs,"fontlab",None)
    fontl = utils.return_key(kwargs,"fontl",None)
    z_binsize = utils.return_key(kwargs,"mark_size",0.2)

    if velunits and kmax==2:
        kmax=0.035
    if velunits and kmin==4e-2:
        kmin=8e-4
    if comparemodel is not None:
        comparemodel_name = comparemodel
        comparemodel = load_model(comparemodel_name,model_file=model_file)
        zmodel,kmodel,kpkmodel=comparemodel


    fig,(ax,ax2) = plt.subplots(2,figsize = (8, 8),gridspec_kw=dict(height_ratios=[3,1]),sharex=True)
    if not velunits:
        adjust_fig(fig)


    try:
        ax.fill_betweenx([-1000,1000],[minrescor,minrescor],[maxrescor,maxrescor],color='0.7',zorder=-30)
        ax2.fill_betweenx([-1000,1000],[minrescor,minrescor],[maxrescor,maxrescor],color='0.7',zorder=-30,label='')
    except:
        pass


    if not velunits:
        ax2.set_xlabel(r' k [1/$\AA$]', fontsize = fontt)
    else:
        ax2.set_xlabel(r' k [s/km]', fontsize = fontt)

    if not plot_P:
        ax.set_ylabel(r'$\Delta^2_{1d}$ ', fontsize=fontt, labelpad=-1)
        ax2.set_ylabel(r'$\Delta^2_{1d,data}-\Delta^2_{1d,DR14fit})/\Delta^2_{1d,DR14fit}$')
    else:
        ax.set_ylabel(r'$P_{1d}$ ', fontsize=fontt, labelpad=-1)
        ax2.set_ylabel(r'$(P_{1d,data}-P_{1d,DR14fit})/P_{1d,DR14fit}$')
    ax.set_yscale('log')

    for a in ax,ax2:
        a.xaxis.set_ticks_position('both')
        a.xaxis.set_tick_params(direction='in')
        a.yaxis.set_ticks_position('both')
        a.yaxis.set_tick_params(direction='in')
        a.xaxis.set_tick_params(labelsize=fontlab)
        a.yaxis.set_tick_params(labelsize=fontlab)
        a.set_xlim(kmin,kmax)

    if not plot_P:
        ax.set_ylim(4e-3,2)
    else:
        if not velunits:
            ax.set_ylim(0.01,0.5)
        else:
            ax.set_ylim(1,300)
    ax2.set_ylim(-diffrange/2,diffrange/2)
    handles, labels = ax.get_legend_handles_labels()

    legend1 = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.03, 0.98), borderaxespad=0.,fontsize = fontl)

    if data is not None:
        ax.errorbar([0],[0], yerr =[0], fmt = 'o',color='k', markersize = mark_size, label ='{}'.format(reslabel))

    if comparison is not None:
        ax.fill_between([0],[0],[0],color='k',label=reslabel2)

    if comparemodel is not None:
        if comparemodel_name == "eBOSSmodel_stack" :
            ax.plot([0],[0],label='eBOSS DR14 fits',color='k',ls=':')
        if comparemodel_name == "DR9model_stack":
            ax.plot([0],[0],label='BOSS DR9 fits',color='k',ls=':')

    handles, labels = ax.get_legend_handles_labels()
    handles,labels=zip(*[(h,l) for (h,l) in zip(handles,labels) if not 'z =' in l])
    ax2.legend(handles, labels, loc=3, bbox_to_anchor=(1.03, 0.02), borderaxespad=0.,fontsize = fontl)


    if not velunits:
        par1.set_xlim(*ax2.get_xlim())
        par2.set_xlim(*ax2.get_xlim())
        par3.set_xlim(*ax2.get_xlim())


    ax.add_artist(legend1)
    fig.subplots_adjust(top=0.95,bottom=0.114,left=0.078,right=0.758,hspace=0.2,wspace=0.2)
    fig.tight_layout()
    reslabel=reslabel.replace('\n','')
    reslabel2=reslabel2.replace('\n','')
    fig.savefig(outname+f"{'' if not plot_P else '_powernotDelta'}_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}.pdf")


    if np.any(diff_data_model):
        plt.figure()
        sns.violinplot(data=pandas.DataFrame(np.array(diff_data_model).T,None,zbins),inner=None,orient='v',palette=colors,scale='width')
        for i,d in enumerate(diff_data_model):
            plt.errorbar(i,np.mean(d),scipy.stats.sem(d,ddof=0), color='0.3',marker='.')
        plt.xlabel('z')
        plt.ylabel('$(P-P_{model})/P$')
        plt.savefig(outname+f"_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}_diff.pdf")
        plt.figure()
        sns.violinplot(data=pandas.DataFrame(np.array(chi_data_model).T,None,zbins),inner=None,orient='v',palette=colors,scale='width')
        for i,d in enumerate(chi_data_model):
            plt.errorbar(i,np.mean(d),scipy.stats.sem(d,ddof=0), color='0.3',marker='.')
        plt.xlabel('z')
        plt.ylabel('$(P-P_{model})/\sigma_P}$')
        plt.savefig(outname+f"_kmax_{kmax}_{reslabel.replace(' ','-').replace('(','').replace(')','')}_{reslabel2.replace(' ','-').replace('(','').replace(')','')}_chi.pdf")





def place_k_speed_unit_axis(fig,ax,fontt=None):
    #this createss more x-axes to compare things in k[s/km]
#    fig.subplots_adjust(top=20.75)
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
#    fig.subplots_adjust(top=20.75)
    par1 = ax.twiny()
    par1.set_xlabel(r' k [s/km] @ z=2.2', fontsize = fontt)
    par1.xaxis.set_major_formatter(FuncFormatter(partial(kskmtokAA,z)))


def compute_and_plot_mean_z_noise_power(data,zbins,outdir="./",fname="noise",kmin=4e-2,kmax=2.5,velunits=False):
    dict_noise_diff=compute_mean_z_noise_power(data,zbins,kmin=kmin,kmax=kmax,velunits=velunits)
    plot_mean_z_noise_power(dict_noise_diff,zbins,outdir=outdir,fname=fname)

def compute_mean_z_noise_power(data,zbins,kmin=4e-2,kmax=2.5,velunits=False):
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




def plot_mean_z_noise_power(dict_noise_diff,zbins,dreshift = 0.02,outdir="./",fname="noise"):
    figure_file = os.path.join(outdir,fname)
    fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
    ax[1].set_xticks(zbins)
    for i in range(len(dict_noise_diff["pipeline"])):
        scale_fac = 3e5/((1+zbins[i])*1216)
        pipeline = dict_noise_diff["pipeline"][i] * scale_fac
        error_pipeline = dict_noise_diff["error_pipeline"][i] * scale_fac
        diff = dict_noise_diff["diff"][i] * scale_fac
        error_diff = dict_noise_diff["error_diff"][i] * scale_fac
        # diff_over_pipeline = dict_noise_diff["diff_over_pipeline"][i]
        # error_diff_over_pipeline = dict_noise_diff["error_diff_over_pipeline"][i]
        diff_over_pipeline = ( dict_noise_diff["diff"][i] - dict_noise_diff["pipeline"][i] ) * scale_fac
        error_diff_over_pipeline = diff_over_pipeline * np.sqrt((error_pipeline/pipeline)**2 + (error_diff/diff)**2)
        ax[0].errorbar(zbins[i],pipeline,error_pipeline,marker='*',color =f"C{i}")
        ax[0].errorbar(zbins[i] + dreshift,diff,error_diff,marker='o',color =f"C{i}")
        ax[1].errorbar(zbins[i],diff_over_pipeline,error_diff_over_pipeline,marker='.')
    legend_elements = [Line2D([], [], color='k', marker='*', linestyle='None', label='$P_{pipeline}$'),
                       Line2D([], [], color='k', marker='o', linestyle='None', label='$P_{diff}$')]
    ax[0].legend(handles=legend_elements)
    ax[0].set_ylabel('$<P> [km/s]$')
    # ax[0].set_ylabel('$mean_{k}(P_{diff}) [\AA]$')
    ax[1].set_ylabel('$\delta<P> [km/s]$')
    ax[1].set_xlabel('z')
    ax[1].legend(handles = [Line2D([], [], color='k', marker='None', linestyle='None',
                                  label='Average for all redshift = ${}$%'.format(np.round(np.mean(dict_noise_diff["diff_over_pipeline"])*100,2)))],frameon=False)
    fig.savefig("{}_mean_ratio_diff_pipeline_power.pdf".format(figure_file),format="pdf")
    fig.savefig("{}_mean_ratio_diff_pipeline_power.png".format(figure_file),format="png",dpi=300)



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
    fig.savefig("{}_mean_ratio_diff_pipeline_power.pdf".format(nameout),format="pdf")



# CR - rewrite with scipy func

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



# CR - reformat function into 3 func

def plot_noise_power_ratio(data,
                           zbins,
                           outdir="./",
                           out_name="P1d",
                           kmin=None,
                           kmax=None,
                           noisediff=False,
                           noise_comparison=False,
                           side_band=False,
                           side_band_comp=None,
                           side_band_legend=["SB1","SB2"],
                           fontt=None,
                           k_units="A"):
    if(k_units == "A"):
        True
    elif(k_units == "kms"):
        data = convert_data_to_kms(data)
    else:
        raise ValueError("choose units between A and kms")


    mean_dict = return_mean_z_dict(zbins,data)

    if(noise_comparison):
        fig2,ax2=plt.subplots(4,1,figsize=(8,10),sharex=True)
    if(side_band):
        fig3,ax3=plt.subplots(4,1,figsize=(8,10),sharex=True)


    fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)
    if(noisediff):
        noise_to_plot,labelnoise = 'meanPk_diff','diff'
    else:
        noise_to_plot,labelnoise = 'meanPk_noise','pipeline'
    for z,d in zip(zbins,data):
        ax[0].plot(d['meank'],d['meanPk_raw'],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[0].set_ylabel('$P_{raw} [\AA]$')
        elif(k_units == "kms"):
            ax[0].set_ylabel('$P_{raw} [km/s]$')
        ax[0].legend()
        if(side_band):
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
        ax[1].plot(d['meank'],d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [\AA]$')
        elif(k_units == "kms"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [km/s]$')
        ax[2].plot(d['meank'],d[noise_to_plot]/d['meanPk_raw'],label=f'{z:.1f}')
        ax[2].set_ylabel('$P_{' + labelnoise +'}/P_{raw}$')
        if(noise_comparison):
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



    if(noise_comparison):
        ax2[3].errorbar(mean_dict["k_array"],(mean_dict["meanPk_diff"] - mean_dict["meanPk_noise"]) / mean_dict["meanPk_noise"], yerr =mean_dict["error_diffovernoise"], fmt = 'o')#,marker_size=6)
        ax2[3].set_ylabel('$mean_{z}((P_{diff} - P_{pipeline})/P_{pipeline})$')
        if(k_units == "A"):
            ax2[3].set_xlabel('k[1/$\AA$]')
            place_k_speed_unit_axis(fig2,ax2[0])
        elif(k_units == "kms"):
            ax2[3].set_xlabel('k[$s/km$]')
        if(kmin is not None): ax2[0].set_xlim(kmin,kmax)
        fig2.tight_layout()
        fig2.savefig(os.path.join(outdir,"{}_ratio_diff_pipeline_power_unit{}.pdf".format(out_name,k_units)),format="pdf")



    if(side_band):
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
#       # ax3[3].plot(cont_k_array,polynome)
#         if(side_band_comp is not None):
#             yerr =np.sqrt( side_band_comp["error_meanPk_noise"]**2 + side_band_comp["error_meanPk_raw"]**2)
#             ax3[3].errorbar(side_band_comp["k_array"],side_band_comp["meanPk_raw"] - side_band_comp["meanPk_noise"],yerr, fmt = 'o',label=side_band_legend[1])
# #            ax3[3].plot(side_band_comp["k_cont"],side_band_comp["poly"])
#             ax3[3].legend()
        if(k_units == "A"):
            ax3[3].set_xlabel('k[1/$\AA$]')
            place_k_speed_unit_axis(fig3,ax3[0])
        elif(k_units == "kms"):
            ax3[3].set_xlabel('k[$s/km$]')
        if(kmin is not None): ax3[0].set_xlim(kmin,kmax)
        fig3.tight_layout()
        fig3.savefig(os.path.join(outdir,"{}_side_band_unit{}.pdf".format(out_name,k_units)),format="pdf")

    ax[3].errorbar(mean_dict["k_array"],mean_dict[noise_to_plot]/mean_dict["meanPk_raw"], yerr =mean_dict["error_{}overraw".format(noise_to_plot)], fmt = 'o')#,marker_size=6)
    ax[3].set_ylabel('$mean_{z}(P_{' + labelnoise +'}/P_{raw})$')
    if(side_band is False):
        try :
            f_exp = lambda x,a,b,c : a
            arg_func = scipy.optimize.curve_fit(f_exp,mean_dict["k_array"],mean_dict[noise_to_plot]/mean_dict["meanPk_raw"],p0=[1,1,1])
            if(np.min(mean_dict["k_array"])> 3.0) : kmin_fit = np.min(mean_dict["k_array"])
            else : kmin_fit = 3.0
            cont_k_array = np.linspace(kmin_fit,np.max(mean_dict["k_array"]),500)
            fit_exp = f_exp(cont_k_array,*arg_exp[0])
            fit_exp_min = f_exp(cont_k_array,*(arg_exp[0]-np.diag(arg_exp[1])))
            fit_exp_max = f_exp(cont_k_array,*(arg_exp[0]+np.diag(arg_exp[1])))
            ax[3].fill_between(cont_k_array, fit_exp_min, fit_exp_max,facecolor='grey', interpolate=True,alpha=0.5)
            ax[3].plot(cont_k_array,fit_exp)
            ax[3].text(np.min(mean_dict["k_array"]),(np.max(mean_dict[noise_to_plot]/mean_dict["meanPk_raw"])+np.min(mean_dict[noise_to_plot]/mean_dict["meanPk_raw"]))/2,"asymptote = {}".format(np.round(arg_exp[0][0],3)))
        except:
            print("Pdiff over Praw fit did not converge")
    if(k_units == "A"):
        ax[3].set_xlabel('k[1/$\AA$]')
        place_k_speed_unit_axis(fig,ax[0])
    elif(k_units == "kms"):
        ax[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax[0].set_xlim(kmin,kmax)
    # ax[0].set_ylim((0,1.25))
    # ax[1].set_ylim((0.05,0.2))
    # ax[2].set_ylim((0,1.3))
    # ax[3].set_ylim((0,1.3))
    for i in [0,1,2,3]:
        ax[i].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,f"{out_name}_ratio_{labelnoise}_raw_power_unit{k_units}.pdf"),format="pdf")
