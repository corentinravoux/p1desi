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
from scipy.stats import binned_statistic





def compute_Pk_means_parallel(data_dir,
                              args,
                              zbins,
                              searchstr='*',
                              ncpu=1,
                              overwrite=False,
                              velunits=False,
                              debug=False,
                              nomedians=False,
                              logsample=False,
                              ):
    outfilename=data_dir+f'mean_Pk1d_snrcut{args["SNR_min"]}_par{"_log" if logsample else ""}.fits.gz'
    if os.path.exists(outfilename) and not overwrite:
        print(f"found existing power, loading from file {outfilename}")
        outdir=t.Table.read(outfilename)
        return outdir
    if ncpu>1:
        from multiprocessing import Pool
    files = glob.glob("{}/Pk1D{}.fits.gz".format(data_dir,searchstr))
    #generate arrays
    zbinedges=zbins-0.1
    zbinedges=np.concatenate([zbinedges,zbinedges[[-1]]+args['z_binsize']])
    if velunits:
        k_inf=args["k_inf_vel"]
        k_sup=args["k_sup_vel"]
        k_dist=args["k_bin_dist_vel"]
    else:
        k_inf=args["k_inf_lin"]
        k_sup=args["k_sup_lin"]
        k_dist=args["k_bin_dist_lin"]
    if not logsample:
        kbinedges=np.arange(k_inf,k_sup,k_dist)
    else:
        kbinedges=10**np.arange(np.log10(k_inf),np.log10(k_sup),0.05)
    outdir=t.Table()
    if ncpu>1:
        with Pool(ncpu) as pool:
            dataarr_all=pool.starmap(compute_single_means,[[f,args,zbinedges,kbinedges,debug,nomedians,logsample] for f in files])
    else:
        dataarr_all=[compute_single_means(f,args,zbinedges,kbinedges,debug=debug,nomedians=nomedians,logsample=logsample) for f in files]
    dataarr_all=[d for d in dataarr_all if d is not None] #filter for files where S/N criterion is never fulfilled
    outdir['N']=np.sum([d['N'] for d in dataarr_all],axis=0)
    outdir['N_chunks']=np.sum([d['N_chunks'] for d in dataarr_all],axis=0)
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
        elif c.startswith('median'):
            #this is only approximate, should be ok if the number of files processed is very large, it's also mem inefficient if that will ever be important (could be improved by doing a smart sorting instead...)
            #dall=[d for d1 in dataarr_all for d in (d1['N']*[d1[c]])]
            #np.nanmedian(dall)
            #this implementation should be faster and more mem efficient, but harder to read and probably not completely right
            dall=np.array([d[c] for d in dataarr_all])
            Nall=np.array([d['N'] for d in dataarr_all])
            dargsort=np.argsort(dall,axis=0)
            dsort=np.take_along_axis(dall, dargsort, axis=0)
            Nsort=np.take_along_axis(Nall, dargsort, axis=0)
            cumsums=np.cumsum(dsort*Nsort,axis=0)
            totsums=np.nansum(dsort*Nsort,axis=0)
            Nsums=np.cumsum(Nsort,axis=0)
            fracs=cumsums/totsums[np.newaxis,...]
            ind50s=np.argmax(fracs>0.5,axis=0)
            #this takes the 2 values closest to the median and does some interpolation
            outdir[c]=np.array([[((1-fracs[ind50s[i,j],i,j])*dsort[ind50s[i,j],i,j]+fracs[ind50s[i,j],i,j]*dsort[ind50s[i,j]-1,i,j]) for j in range(dall.shape[2])] for i in range(dall.shape[1])])
        elif c.startswith('max'):
            outdir[c]=np.nanmax([d[c] for d in dataarr_all],axis=0)
        elif c.startswith('min'):
            outdir[c]=np.nanmin([d[c] for d in dataarr_all],axis=0)
    if logsample:
        outdir.meta['LOGBINS']=1
    else:
        outdir.meta['LOGBINS']=0
    outdir.write(outfilename,overwrite=overwrite)
    return outdir




def compute_single_means(f,
                         args,
                         zbinedges=None,
                         kbinedges=None,
                         debug=False,
                         nomedians=False,
                         logsample=False):
    dataarr=[]
    outdir=t.Table()
    zarr=[]
    with fitsio.FITS(f) as hdus:
        for i,h in enumerate(hdus[1:]):
            data=h.read()
            header=h.read_header()
            tab=t.Table(data)
            tab['z']=float(header['MEANZ'])
            tab['snr']=float(header['MEANSNR'])
            if float(header['meansnr'])<args['SNR_min']:
                continue
            if (tab['Pk_noise'][tab['k']<kbinedges[-1]]>tab['Pk_raw'][tab['k']<kbinedges[-1]]*10000000).any():
                print(f"file {f} hdu {i+1} has very high noise power, ignoring, max value: {(tab['Pk_noise'][tab['k']<kbinedges[-1]]/tab['Pk_raw'][tab['k']<kbinedges[-1]]).max()}*Praw")
                continue
            dataarr.append(tab)
            zarr.append(float(header['MEANZ']))
    #this part could be done per file for larger datasets and then recombined after
    if len(dataarr)<1:
        print(f"only {len(dataarr)} spectra in file, ignoring this as it currently messes with analysis")
        return None
    dataarr=t.vstack(dataarr)
    dataarr['Delta2']=dataarr['k']*dataarr['Pk']/np.pi
    dataarr['Pk_norescor']=dataarr['Pk_raw']-dataarr['Pk_noise']
    dataarr['Pk/Pnoise']=dataarr['Pk_raw']/dataarr['Pk_noise']  #take the ratio this way as the noise power will fluctuate less even in the tails (i.e. less divisions by 0)
    cols=dataarr.colnames
    N,zedges,kedges,numbers=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr['k'],statistic='count',bins=[zbinedges,kbinedges])
    N_chunks,zedges_chunks,numbers_chunks=binned_statistic(zarr,zarr,statistic='count',bins=zbinedges)
    for c in cols:
        outdir['mean'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='mean',bins=[zedges, kedges])
        if not nomedians:
            outdir['median'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='median',bins=[zedges, kedges])
        outdir['error'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='std',bins=[zedges, kedges])
        outdir['min'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='min',bins=[zedges, kedges])
        outdir['max'+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic='max',bins=[zedges, kedges])
        outdir['error'+c]/=np.sqrt(N) #to get the error on the mean instead of standard deviation in the data
    outdir['N']=N
    outdir['N_chunks']=np.array(N_chunks,dtype=int)
    if debug:
        outdir.write(f"{f[:-8]+'_mean'}{'_log' if logsample else ''}.fits.gz",overwrite=True)   #this will be slow because it writes the data for each file
    return outdir





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
                     k_bin_vel=0.002):
    """ old, might be obsolete """
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