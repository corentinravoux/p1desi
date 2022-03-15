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
                              logsample=False
                              ):
    """Computes the power spectrum from picca_pk1d outputs

    Args:
        data_dir ([type]): where the outputs are living
        args ([type]): a dictionary including k range limits and SNR cut
        zbins ([type]): which redshifts to use
        searchstr (str, optional): [description]. Defaults to '*'.
        ncpu (int, optional): Should we multiprocess? Defaults to 1.
        overwrite (bool, optional): Overwrite files if existing. Defaults to False.
        velunits (bool, optional): Compute the power in velocity space (input is still assumed to be in angstrom units) Defaults to False.
        debug (bool, optional): generates intermediate files. Defaults to False.
        nomedians (bool, optional): should we skip computation of medians (faster). Defaults to False.
        logsample (bool, optional): Should the k-bins be sampled in log instead of linearly. Defaults to False.

    Returns:
        [type]: [description]
    """
    outfilename=os.path.join(data_dir,f'mean_Pk1d_par{"_log" if logsample else ""}{"_vel" if velunits else ""}.fits.gz')
    if os.path.exists(outfilename) and not overwrite:
        print(f"found existing power, loading from file {outfilename}")
        outdir=t.Table.read(outfilename)
        return outdir
    if ncpu>1:
        from multiprocessing import Pool
    files = glob.glob(os.path.join(data_dir,f"Pk1D{searchstr}.fits.gz"))
    #generate arrays
    zbinedges=zbins-args['z_binsize']/2
    zbinedges=np.concatenate([zbinedges,zbinedges[[-1]]+args['z_binsize']])
    (k_inf,k_sup,k_dist) = define_wavevector_limits(args,velunits)
    if not logsample:
        kbinedges=np.arange(k_inf,k_sup,k_dist)
    else:
        kbinedges=10**np.arange(np.log10(k_inf),np.log10(k_sup),0.05)
    outdir=t.Table()
    if ncpu>1:
        with Pool(ncpu) as pool:
            dataarr_all=pool.starmap(compute_single_means,[[f,args,zbinedges,kbinedges,debug,nomedians,logsample,velunits] for f in files])
    else:
        dataarr_all=[compute_single_means(f,args,zbinedges,kbinedges,debug=debug,nomedians=nomedians,logsample=logsample,velunits=velunits) for f in files]
    dataarr_all=[d for d in dataarr_all if d is not None] #filter for files where S/N criterion is never fulfilled
    outdir['N']=np.sum([d['N'] for d in dataarr_all],axis=0)
    outdir['N_chunks']=np.sum([d['N_chunks'] for d in dataarr_all],axis=0)
    for c in dataarr_all[0].colnames:
        if c.startswith('mean'):
            outdir[c]=np.nansum([d[c]*d['N'] for d in dataarr_all],axis=0)/outdir['N']
        elif c.startswith('error'):
            outdir[c]=np.sqrt(
                (
                    np.nansum(
                    #the following computes mean(d^2), then subtracts mean(d)^2 which was precomputed, the last division is for getting the error on the mean
                        [d['N']*(d[c]**2+d[c.replace('error','mean')]**2) for d in dataarr_all], axis=0)/
                        outdir['N']-outdir[c.replace('error','mean')]**2)
                /outdir['N'])
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
    outdir.meta['velunits']=velunits
    outdir.write(outfilename,overwrite=overwrite)
    return outdir




def compute_single_means(f,
                         args,
                         zbinedges=None,
                         kbinedges=None,
                         debug=False,
                         nomedians=False,
                         logsample=False,
                         velunits=False):
    dataarr=[]
    outdir=t.Table()
    zarr=[]
    with fitsio.FITS(f) as hdus:
        for i,h in enumerate(hdus[1:]):
            data=h.read()
            header=h.read_header()
            tab=t.Table(data)
            try:
                tab.rename_column('K','k')
                tab.rename_column('PK','Pk')
                tab.rename_column('PK_RAW','Pk_raw')
                tab.rename_column('PK_NOISE','Pk_noise')
                tab.rename_column('PK_DIFF','Pk_diff')
                tab.rename_column('COR_RESO','cor_reso')
            except:
                pass
            if np.nansum(tab['Pk'])==0:
                tab['Pk']=(tab['Pk_raw']-tab['Pk_noise'])/tab['cor_reso']
            tab['z']=float(header['MEANZ'])
            tab['snr']=float(header['MEANSNR'])
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
    dataarr['Pk_nonoise']=dataarr['Pk_raw']/dataarr['cor_reso']
    dataarr['Pk/Pnoise']=dataarr['Pk_raw']/dataarr['Pk_noise']  #take the ratio this way as the noise power will fluctuate less even in the tails (i.e. less divisions by 0)
    cols=dataarr.colnames
    N_chunks,zedges_chunks,numbers_chunks=binned_statistic(zarr,zarr,statistic='count',bins=zbinedges)

    statsarr=['mean','error','min','max']
    if not nomedians:
        statsarr+=['median']
    if not velunits:
        N,zedges,kedges,numbers=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr['k'],statistic='count',bins=[zbinedges,kbinedges])
        for c in cols:
            for stats in statsarr:  #TODO: double check if error becomes 0 or nan if N=1
                outdir[stats+c],_,_,_=binned_statistic_2d(dataarr['z'],dataarr['k'],dataarr[c],statistic=stats if stats!='error' else 'std',bins=[zedges, kedges])
        outdir['N']=N
        outdir['N_chunks']=np.array(N_chunks,dtype=int)
    else:
        outdir=t.Table()
        for izbin,zbin in enumerate(zbinedges[:-1]):
            select=(dataarr['z'][:]<zbinedges[izbin+1])&(dataarr['z'][:]>zbinedges[izbin])
            coldir=t.Table()
            coldir['N_chunks']=np.array([N_chunks[izbin]],dtype=int)
            if N_chunks[izbin]==0:
                coldir['N']=np.zeros((1,len(kbinedges)-1))
                for c in cols:
                    for stats in statsarr:
                        coldir[stats+c]=np.ones((1,len(kbinedges)-1))*np.nan
            else:
                convfactor=(1215.67*(1+np.mean(dataarr['z'][select])))/3e5
                dataarr['k'][select]*=convfactor
                for c in cols:
                    if 'Pk' in c:
                        dataarr[c][select]/=convfactor


                N,kedges,numbers=binned_statistic(dataarr['k'][select],dataarr['k'][select],statistic='count',bins=kbinedges)
                coldir['N']=N[np.newaxis,:]

                for c in cols:
                    for stats in statsarr:
                        st,_,_=binned_statistic(dataarr['k'][select],dataarr[c][select],statistic=stats if stats!='error' else 'std',bins=kedges)
                        coldir[stats+c]=st[np.newaxis,:]
            outdir=t.vstack([outdir,coldir])
    if debug:
        outdir.meta['velunits']=velunits
        outdir.write(f"{f[:-8]+'_mean'}{'_log' if logsample else ''}{'_vel' if velunits else ''}.fits.gz",overwrite=True)   #this will be slow because it writes the data for each file
    return outdir





def compute_Pk_means(data_dir,
                     args,
                     ncpu=1,
                     ignore_existing=False,
                     velunits=False,
                     no_zbinning=False):
    """ old, might be obsolete """
    files = glob.glob(os.path.join(data_dir,"*.fits.gz"))
    # files = glob.glob("{}/*.fits.gz".format(data_dir))
    #generate arrays
    zbinedges=np.arange(2.1,4.7,0.2)
    (k_inf,k_sup,k_dist) = define_wavevector_limits(args,velunits)
    kbinedges=np.linspace(k_inf,k_sup,k_dist)
    outdir=t.Table()
    dataarr=[]
    for f in files:
        hdus=fitsio.FITS(f)
        for h in hdus[1:]:
            data=h.read()
            header=h.read_header()
            tab=t.Table(data)
            try:
                tab.rename_column('K','k')
                tab.rename_column('PK','Pk')
                tab.rename_column('PK_RAW','Pk_raw')
                tab.rename_column('PK_NOISE','Pk_noise')
                tab.rename_column('PK_DIFF','Pk_diff')
                tab.rename_column('COR_RESO','cor_reso')
            except:
                pass
            if np.nansum(tab['Pk'])==0:
                tab['Pk']=(tab['Pk_raw']-tab['Pk_noise'])/tab['cor_reso']
            tab['z']=float(header['MEANZ'])
            tab['snr']=float(header['MEANSNR'])
            dataarr.append(tab)
    #this part could be done per file for larger datasets and then recombined after
    dataarr=t.vstack(dataarr)
    #the following needs to be case insensitive
    dataarr['Delta2']=dataarr['k']*dataarr['Pk']/np.pi
    dataarr['Delta2_nonoise']=dataarr['k']*(dataarr['Pk_raw']/dataarr['cor_reso'])/np.pi
    dataarr['Pk_norescor']=dataarr['Pk_raw']-dataarr['Pk_noise']
    dataarr['Pk_nonoise']=dataarr['Pk_raw']/dataarr['cor_reso']
    dataarr['Pk/Pnoise']=dataarr['Pk_raw']/dataarr['Pk_noise']
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



def define_wavevector_limits(args,velunits):
    pixsize_desi = 0.8
    if velunits:
        if("k_inf_vel" in args.keys()):
            k_inf=args["k_inf_vel"]
            k_sup=args["k_sup_vel"]
            k_dist=args["k_bin_dist_vel"]
        else:
            k_inf=0.000813
            k_dist=0.000542*args["rebinfac"]
            k_inf_lin=2*np.pi/((1200-1050)*(1+3.4)/args["rebinfac"])
            k_sup_lin=np.pi/pixsize_desi
            nb_k_bin=int(k_sup_lin/k_inf_lin/4)
            k_sup=k_inf + nb_k_bin*k_dist
    else:
        if("k_inf_lin" in args.keys()):
            k_inf=args["k_inf_lin"]
            k_sup=args["k_sup_lin"]
            k_dist=args["k_bin_dist_lin"]
        else:

            k_inf=2*np.pi/((1200-1050)*(1+3.4)/args["rebinfac"])
            k_sup=np.pi/pixsize_desi
            nb_k_bin=int(k_sup/k_inf/4)
            k_dist=(k_sup-k_inf)/nb_k_bin
    return(k_inf,k_sup,k_dist)
