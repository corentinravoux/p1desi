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
import sys,os
import numpy as np
import fitsio
import astropy.table as t
from picca.Pk1D import Pk1D
import functools
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

#
#
# def summation_per_file(param,zbins,nb_k_bin,args,k_inf_lin,
#                        k_sup_lin,velunits,k_inf_vel,k_sup_vel,
#                        noisediff=False):
#     nb_z_bin = len(zbins)
#     (i,f)=param
#     if i%10:
#         sys.stderr.write("\rread {}".format(i))
#
#     # read fits files
#     hdus = fitsio.FITS(f)
#     try:
#         pk1ds = [Pk1D.from_fitsio(h) for h in hdus[1:]]
#     except:
#         print('error in file: {}\n'.format(f))
#         return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#     sum_mode = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#
#     kout=np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumPk = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumPk2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumDelta2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumDelta4 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumrescor = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     zout = np.zeros([nb_z_bin],dtype=np.float64)
#     mean_snr = [[] for i in range(nb_z_bin)]
#     sumz = np.zeros([nb_z_bin],dtype=np.float64)
#     sumPk_raw = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumPk_raw2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumPk_noise = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     sumPk_noise2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     if(noisediff):
#         sumPk_diff = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumPk_diff2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#     else :
#         sumPk_diff = None
#         sumPk_diff2 = None
#
#     if velunits:
#         k_inf=k_inf_vel
#         k_sup=k_sup_vel
#     else:
#         k_inf=k_inf_lin
#         k_sup=k_sup_lin
#
#     # loop over pk1ds
#     for pk in pk1ds:
#         # Selection over the SNR and the resolution
#         if (pk.mean_snr<=args['SNR_min'] or pk.mean_reso>=args['reso_max']) : continue
#         iz = np.abs(pk.mean_z-zbins)<args['z_binsize']/2
#         if(iz.any()):
#             iz=iz.nonzero()[0][0]
#         else:
#             continue
#         for i2,ki in enumerate(pk.k) :
#             ik = int((pk.k[i2]-k_inf)/(k_sup-k_inf)*nb_k_bin);
#
#
#             if(ik>= nb_k_bin or ik<0) : continue
#             Delta2 = pk.Pk[i2]*ki/np.pi
#             sumDelta2[iz,ik] += Delta2
#             sumDelta4[iz,ik] += Delta2**2
#             sumPk[iz, ik] += pk.Pk[i2]
#             sumPk2[iz, ik] += pk.Pk[i2]**2
#             sumrescor[iz, ik] += pk.cor_reso[i2]
#             sumPk_raw[iz, ik] += pk.Pk_raw[i2]
#             sumPk_raw2[iz, ik] += pk.Pk_raw[i2]**2
#             sumPk_noise[iz, ik]+= pk.Pk_noise[i2]
#             sumPk_noise2[iz, ik]+= pk.Pk_noise[i2]**2
#             if(noisediff):
#                 sumPk_diff[iz, ik]+= pk.Pk_diff[i2]
#                 sumPk_diff2[iz, ik]+= pk.Pk_diff[i2]**2
#             kout[iz,ik] += ki
#             sum_mode[iz,ik] += 1.0
#
#         mean_snr[iz].append(pk.mean_snr)
#         zout[iz]+=pk.mean_z
#         sumz[iz]+=1
#     for i in range(len(mean_snr)):
#         mean_snr[i] = np.mean(mean_snr[i])
#     return (sumDelta2, sumDelta4, sumPk, sumPk2, sumrescor,
#            kout, sum_mode, zout, sumz,sumPk_raw,sumPk_raw2,
#            sumPk_noise,sumPk_noise2,sumPk_diff,sumPk_diff2)
#
#
#
# def compute_Pk_means_old(data_dir,zbins,nb_k_bin,args,k_inf_lin=0.01,
#                          k_sup_lin=10.0,k_inf_vel=0.00001,k_sup_vel=0.1,
#                          ncpu=1,ignore_existing=False,velunits=False,
#                          noisediff=False):
#
#     nb_z_bin = len(zbins)
#     try:
#         if ignore_existing:
#             raise IndexError
#         fname=glob.glob(data_dir+'/mean_Pk*.fits')
#         output=t.Table.read(fname[0])
#     except IndexError:
#         fi = glob.glob("{}/*.fits.gz".format(data_dir))
#         print("Couldn't read mean table, recomputing")
#
#         sumPk = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumPk2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumDelta2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumDelta4 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumrescor = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         zout = np.zeros([nb_z_bin],dtype=np.float64)
#         sumz=np.zeros([nb_z_bin],dtype=np.float64)
#         sum_mode = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         kout=np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumPk_raw = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumPk_raw2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumPk_noise = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         sumPk_noise2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         if(noisediff):
#             sumPk_diff = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#             sumPk_diff2 = np.zeros([nb_z_bin,nb_k_bin],dtype=np.float64)
#         else:
#             sumPk_diff = None
#             sumPk_diff2 = None
#
#
#         param=list(enumerate(fi))
#         func=functools.partial(summation_per_file,zbins=zbins,nb_k_bin=nb_k_bin,
#                                args=args,k_inf_lin=k_inf_lin,k_sup_lin=k_sup_lin,
#                                velunits=velunits,k_inf_vel=k_inf_vel,
#                                k_sup_vel=k_sup_vel,noisediff=noisediff)
#         if ncpu==1:
#             out=[]
#             for p in param:
#                 out.append(func(p))
#
#         else:
#             import multiprocessing
#             pool=multiprocessing.Pool(ncpu)
#             out=pool.map(func,param)
#         for (Delta2, Delta4, Pk, Pk2, rescor, ko, su, zo, z, s_Pk_raw,s_Pk_raw2,s_Pk_noise,s_Pk_noise2,s_Pk_diff,s_Pk_diff2) in out:
#                 sumDelta2+=Delta2
#                 sumDelta4+=Delta4
#                 sumPk+=Pk
#                 sumPk2+=Pk2
#                 sumrescor+=rescor
#                 zout+=zo
#                 kout+=ko
#                 sum_mode+=su
#                 sumz+=z
#                 sumPk_raw+=s_Pk_raw
#                 sumPk_raw2+=s_Pk_raw2
#                 sumPk_noise+=s_Pk_noise
#                 sumPk_noise2+=s_Pk_noise2
#                 if(noisediff):
#                     sumPk_diff+=s_Pk_diff
#                     sumPk_diff2+=s_Pk_diff2
#
#         #Compute means and errors
#         meanDelta2 = np.where(sum_mode!=0,sumDelta2/sum_mode,0.0)
#         errorDelta2 = np.where(sum_mode!=0,np.sqrt(((sumDelta4/sum_mode)-meanDelta2**2)/(sum_mode-1)),0.0)
#         meanPk = np.where(sum_mode!=0,sumPk/sum_mode,0.0)
#         errorPk = np.where(sum_mode!=0,np.sqrt(((sumPk2/sum_mode)-meanPk**2)/sum_mode),0.0)
#         meank = np.where(sum_mode!=0,kout/sum_mode,0.0)
#         meanrescor = np.where(sum_mode!=0,sumrescor/sum_mode,0.0)
#         zout=np.where(sumz!=0,zout/sumz,0.0)
#         meanPk_raw = np.where(sum_mode!=0,sumPk_raw/sum_mode,0.0)
#         errorPk_raw = np.where(sum_mode!=0,np.sqrt(((sumPk_raw2/sum_mode)-meanPk_raw**2)/sum_mode),0.0)
#         meanPk_noise = np.where(sum_mode!=0,sumPk_noise/sum_mode,0.0)
#         errorPk_noise = np.where(sum_mode!=0,np.sqrt(((sumPk_noise2/sum_mode)-meanPk_noise**2)/sum_mode),0.0)
#         if(noisediff):
#             meanPk_diff = np.where(sum_mode!=0,sumPk_diff/sum_mode,0.0)
#             errorPk_diff = np.where(sum_mode!=0,np.sqrt(((sumPk_diff2/sum_mode)-meanPk_diff**2)/sum_mode),0.0)
#
#         output=t.Table()
#         output['z']=zout
#
#         output['k']=meank
#         output['meanDelta2']=meanDelta2
#         output['errorDelta2']=errorDelta2
#         output['meanPk']=meanPk
#         output['errorPk']=errorPk
#         output['meanPk_raw']=meanPk_raw
#         output['errorPk_raw']=errorPk_raw
#         output['meanPk_noise']=meanPk_noise
#         output['errorPk_noise']=errorPk_noise
#         if(noisediff):
#             output['meanPk_diff']=meanPk_diff
#             output['errorPk_diff']=errorPk_diff
#
#         output['rescor']=meanrescor
#         output['nmodes']=sum_mode
#         output['nspec']=sumz
#
#     return output
