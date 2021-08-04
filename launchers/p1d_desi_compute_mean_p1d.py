#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""

import numpy as np
from p1d_desi import p1d_treat


#### COMPUTE PK MEAN ARGS #####

zbins = np.array([ 2.2, 2.4, 2.6 , 2.8, 3.0,3.2,3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])


rebinfac=1
pixsize=0.8
k_inf_lin=2*np.pi/((1200-1050)*(1+3.4)/rebinfac)
k_sup_lin=np.pi/pixsize
nb_k_bin=int(k_sup_lin/k_inf_lin/4)
k_bin_dist_lin=(k_sup_lin-k_inf_lin)/nb_k_bin

k_inf_vel=0.000813
k_bin_dist_vel=0.000542*rebinfac
k_sup_vel=k_inf_vel + nb_k_bin*k_bin_dist_vel

ncpu=4
overwrite=True
velunits=False
debug=False
nomedians=False
logsample=False
searchstr = '*'

args={}
# args['reso_max']=85.
args['SNR_min']=1
args['z_binsize']=0.2
args['k_inf_lin']=k_inf_lin
args['k_sup_lin']=k_sup_lin
args['k_bin_dist_lin']=k_bin_dist_lin
args['k_inf_vel']=k_inf_lin
args['k_sup_vel']=k_sup_lin
args['k_bin_dist_vel']=k_bin_dist_lin

path_to_pk = "/local/home/cravoux/Documents/desi_p1d_development/example_data/everest_sv1only_snr4"



if __name__ == "__main__":


    data=p1d_treat.compute_Pk_means_parallel(path_to_pk,
                                             args,
                                             zbins,
                                             searchstr=searchstr,
                                             ncpu=ncpu,
                                             overwrite=overwrite,
                                             velunits=velunits,
                                             debug=debug,
                                             nomedians=nomedians,
                                             logsample=logsample)
