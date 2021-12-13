#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""

import numpy as np
from p1d_desi import treatpk
import sys

path_to_pk = str(sys.argv[1])
velunits = bool(str(sys.argv[2]) == "True")


#### COMPUTE PK MEAN ARGS #####

zbins = np.array([ 2.2, 2.4, 2.6 , 2.8, 3.0,3.2,3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])


rebinfac=1
pixsize=0.8
k_inf_lin=2*np.pi/((1200-1050)*(1+3.4)/rebinfac)
k_sup_lin=np.pi/pixsize
nb_k_bin=int(k_sup_lin/k_inf_lin/4)
k_bin_dist_lin=(k_sup_lin-k_inf_lin)/nb_k_bin

k_inf_vel=0.000813
k_bin_dist_vel=0.000542*rebinfac
k_sup_vel=k_inf_vel + nb_k_bin*k_bin_dist_vel

ncpu=8
overwrite=False
debug=False
nomedians=False
logsample=False
searchstr = '*'
args={}
args['z_binsize']=0.2
args['k_inf_lin']=k_inf_lin
args['k_sup_lin']=k_sup_lin
args['k_bin_dist_lin']=k_bin_dist_lin
args['k_inf_vel']=k_inf_vel
args['k_sup_vel']=k_sup_vel
args['k_bin_dist_vel']=k_bin_dist_vel




if __name__ == "__main__":
    print("Treating path: ",path_to_pk)
    data=treatpk.compute_Pk_means_parallel(path_to_pk,
                                             args,
                                             zbins,
                                             searchstr=searchstr,
                                             ncpu=ncpu,
                                             overwrite=overwrite,
                                             velunits=velunits,
                                             debug=debug,
                                             nomedians=nomedians,
                                             logsample=logsample)
