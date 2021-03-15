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

import glob
import os
import numpy as np
import fitsio
import astropy.table as t
from scipy.stats import binned_statistic_2d





#### FUNCTIONS #####

def compute_Pk_means(data_dir,
                     ncpu=1,
                     ignore_existing=False,
                     velunits=False,
                     no_zbinning=False,
                     k_inf_lin=0.01,
                     k_sup_lin=10.0,
                     k_bin_lin = 0.2,
                     k_inf_vel=0.00001,
                     k_sup_vel=0.1,
                     k_bin_vel=0.002,
                     noisediff=False):
    files = glob.glob(os.path.join(data_dir,"*.fits.gz"))
    # files = glob.glob("{}/*.fits.gz".format(data_dir))
    #generate arrays
    zbinedges=np.arange(2.1,4.7,0.2)
    if velunits:
        k_inf=k_inf_vel
        k_sup=k_sup_vel
        k_bin=k_bin_vel
    else:
        k_inf=k_inf_lin
        k_sup=k_sup_lin
        k_bin=k_bin_lin
    kbinedges=np.linspace(k_inf,k_sup,k_bin)
    outdir=t.Table()
    dataarr=[]
    for f in files:
        hdus=fitsio.FITS(f)
        for h in hdus[1:]:
            data=h.read()
            header=h.read_header()
            tab=t.Table(data)
            tab['z']=float(header['meanz'])
            tab['snr']=float(header['meansnr'])
            dataarr.append(tab)
    #this part could be done per file for larger datasets and then recombined after
    dataarr=t.vstack(dataarr)
    dataarr['Delta2']=dataarr['k']*dataarr['Pk']/np.pi
    cols=dataarr.colnames
    N,zedges,kedges,numbers=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr['k'],statistic='count',bins=[zbinedges,kbinedges])
    for c in cols:
        outdir['mean'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='mean',bins=[zedges, kedges])
        outdir['median'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='median',bins=[zedges, kedges])
        outdir['error'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='std',bins=[zedges, kedges])
        outdir['min'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='min',bins=[zedges, kedges])
        outdir['max'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='max',bins=[zedges, kedges])
        outdir['error'+c]/=np.sqrt(N) #to get the error on the mean instead of standard deviation in the data
    outdir['N']=N
    return(outdir)



def compute_single_means(f,zbinedges=None,kbinedges=None,debug=False):
    dataarr=[]
    outdir=t.Table()
    with fitsio.FITS(f) as hdus:
        for h in hdus[1:]:
            data=h.read()
            header=h.read_header()
            tab=t.Table(data)
            tab['z']=float(header['MEANZ'])
            tab['snr']=float(header['MEANSNR'])
            if float(header['meansnr'])<args['SNR_min']:
                continue
            dataarr.append(tab)
    #this part could be done per file for larger datasets and then recombined after
    if len(dataarr)==0:
        return None
    dataarr=t.vstack(dataarr)
    dataarr['Delta2']=dataarr['k']*dataarr['Pk']/np.pi
    dataarr['noise_power_ratio']=dataarr['Pk_noise']/dataarr['Pk_raw']
    cols=dataarr.colnames
    N,zedges,kedges,numbers=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr['k'],statistic='count',bins=[zbinedges,kbinedges])
    for c in cols:
        outdir['mean'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='mean',bins=[zedges, kedges])
        #outdir['median'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='median',bins=[zedges, kedges])
        outdir['error'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='std',bins=[zedges, kedges])
        outdir['min'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='min',bins=[zedges, kedges])
        outdir['max'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='max',bins=[zedges, kedges])
        outdir['error'+c]/=np.sqrt(N) #to get the error on the mean instead of standard deviation in the data
    outdir['N']=N
    if debug:
        outdir.write(f"{f[:-8]+'_mean.fits.gz'}",overwrite=True)   #this will be slow because it writes the data for each file
    return(outdir)



def compute_Pk_means_parallelizible(data_dir,ncpu=1,overwrite=False,velunits=False,debug=False):
    if ncpu>1:
        from multiprocessing import Pool
    files = glob.glob("{}/Pk1D*.fits.gz".format(data_dir))
    #generate arrays
    zbinedges=np.arange(2.1,3.5,0.2)
    if velunits:
        k_inf=k_inf_vel
        k_sup=k_sup_vel
        k_dist=k_bin_dist_vel
    else:
        k_inf=k_inf_lin
        k_sup=k_sup_lin
        k_dist=k_bin_dist_lin

    kbinedges=np.arange(k_inf,k_sup,k_dist)

    outdir=t.Table()
    if ncpu>1:
        with Pool(ncpu) as pool:
            dataarr_all=pool.starmap(compute_single_means,[[f,zbinedges,kbinedges,debug] for f in files])
    else:
        dataarr_all=[compute_single_means(f,zbinedges,kbinedges,debug=debug) for f in files]
    dataarr_all=[d for d in dataarr_all if d is not None] #filter for files where S/N criterion is never fulfilled

    outdir['N']=np.sum([d['N'] for d in dataarr_all],axis=0)
    for c in dataarr_all[0].colnames:
        if c.startswith('mean'):
            outdir[c]=np.nansum([d[c]*d['N'] for d in dataarr_all],axis=0)/outdir['N']
        elif c.startswith('error'):
            outdir[c]=np.sqrt(
                (np.nansum(
                #the following computes mean(d^2), then subtracts mean(d)^2 which was precomputed, the last division is for getting the error on the mean
                [d['N']*(d['N']*d[c]**2+d[c.replace('error','mean')]**2) for d in dataarr_all],
                axis=0)/outdir['N']
                               -outdir[c.replace('error','mean')]**2)
                /(outdir['N']))


        elif c.startswith('max'):
            outdir[c]=np.nanmax([d[c] for d in dataarr_all],axis=0)
        elif c.startswith('min'):
            outdir[c]=np.nanmin([d[c] for d in dataarr_all],axis=0)
    outdir.write(data_dir+'mean_Pk1d_par.fits.gz',overwrite=overwrite)

    return (outdir)
