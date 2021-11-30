#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""

import numpy as np
from p1d_desi import p1d_plot
import sys,os

### Input


path_to_pk = str(sys.argv[1])
path_plot = str(sys.argv[2])
region_sb = str(sys.argv[3])
path_sb_substract = str(sys.argv[4])
substract_sb = False
if(region_sb != "None"):
    substract_sb = True
velunits = bool(str(sys.argv[5]) == "True")
model_compare = str(sys.argv[6])
zmax = float(sys.argv[7])
diff_study = bool(str(sys.argv[8]) == "True")
comparison_str = str(sys.argv[9])

### P1D

if model_compare == "None":
    plot_diff_model = False
    comparison = None
    comparison_model = None
    model_compare_str = "None"
else:
    plot_diff_model = True
    if(comparison_str == "None"):
        comparison = None
        comparison_model = model_compare
        model_compare_str = model_compare
    else:
        comparison = model_compare
        comparison_model = None
        model_compare_str = comparison_str




pk_means_name = os.path.join(path_to_pk,
                             f"mean_Pk1d_snrcut1_par{'_vel' if velunits else ''}.fits.gz")
if(substract_sb):
    pk_means_name_sb = os.path.join(path_sb_substract,
                                    f"mean_Pk1d_snrcut1_par{'_vel' if velunits else ''}.fits.gz")


outname = os.path.join(f"{path_plot}_model{model_compare_str}_zmax{zmax}_unit{'kms' if velunits else 'A'}")

comparison_model_file = ["/global/homes/r/ravouxco/2_Software/Python/Data/p1d_models/models_eBOSS_lowz.fits",
                         "/global/homes/r/ravouxco/2_Software/Python/Data/p1d_models/models_eBOSS_highz.fits"]
plot_P = False

k_inf_lin = 4e-2
k_sup_lin = 2.5
k_inf_vel = 0.000813
k_sup_vel = 0.056

zbins = []
z = 2.2
while(z <= zmax):
    zbins.append(z)
    z = z + 0.2
zbins = np.array(zbins)

if(velunits):
    kmax = k_sup_vel
    kmin = k_inf_vel
else:
    kmax = k_sup_lin
    kmin = k_inf_lin


kwargs = {"res_label" :  'DESI',
          "res_label2" : model_compare_str,
          "diff_range" : 1.0,
          "no_errors_diff" : True,
          "marker_size" : 5,
          "marker_style" : ".",
          "fonttext" : 12,
          "fontlab" : 11,
          "fontlegend" : 11,
          "z_binsize" : 0.2,
          "kmin" : kmin,
          "kmax" : kmax,
          "grid" : True}


### Noise study mean z

plot_noise_ratio = True

use_diff_noise = False
plot_noise_comparison_mean_k = False
if(diff_study):
    use_diff_noise = True
    plot_noise_comparison_mean_k = True

plot_side_band = False
k_units_noise_study = "A"
fit_asymptote_ratio = True

kwargs_noise1 = {"ncol_legend" : 2,
                 "kmin" : None,
                 "kmax" : None}


### Noise study mean k

plot_noise_comparison_mean_z = False
if(diff_study):
    plot_noise_comparison_mean_z = True
kwargs_noise2 = {"kmin" : kmin,
                 "kmax" : kmax}



if __name__ == "__main__":
    print("Plotting path: ",pk_means_name)
    data = p1d_plot.read_pk_means(pk_means_name)
    p1d_plot.plot_data(data,
                       zbins,
                       outname,
                       plot_P=plot_P,
                       comparison=comparison,
                       comparison_model=comparison_model,
                       comparison_model_file=comparison_model_file,
                       plot_diff=plot_diff_model,
                       **kwargs)

    if(substract_sb):
        pk_means_sb = p1d_plot.read_pk_means(pk_means_name_sb)
        p1d_plot.plot_data(data,
                           zbins,
                           f"{outname}_{region_sb}_substracted",
                           plot_P=plot_P,
                           comparison=comparison,
                           comparison_model=comparison_model,
                           comparison_model_file=comparison_model_file,
                           plot_diff=plot_diff_model,
                           substract_sb=pk_means_sb,
                           **kwargs)



    p1d_plot.plot_noise_study(data,
                              zbins,
                              outname,
                              k_units_noise_study,
                              use_diff_noise,
                              plot_noise_ratio,
                              plot_noise_comparison_mean_k,
                              plot_side_band,
                              side_band_comp=None,
                              side_band_legend=["SB1","SB2"],
                              fit_asymptote_ratio= fit_asymptote_ratio,
                              **kwargs_noise1)


    if(plot_noise_comparison_mean_z):
        p1d_plot.compute_and_plot_mean_z_noise_power(data,
                                                     zbins,
                                                     outname,
                                                     **kwargs_noise2)
