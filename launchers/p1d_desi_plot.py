#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:50:01 2020

@author: cravoux
"""

import numpy as np
from p1d_desi import p1d_plot


### P1D

pk_means_name = "/local/home/cravoux/Documents/desi_p1d_development/example_data/everest_sv1only_snr4/mean_Pk1d_snrcut1_par.fits.gz"
zbins = np.array([ 2.2, 2.4, 2.6 , 2.8, 3.0,3.2,3.4, 3.6, 3.8, 4.0])#, 4.2, 4.4, 4.6])
outname = "test"
comparison = None
comparison_model = "eBOSSmodel_stack"
comparison_model_file = ["/local/home/cravoux/Documents/Python/Data/p1d_models/models_eBOSS_lowz.fits",
                         "/local/home/cravoux/Documents/Python/Data/p1d_models/models_eBOSS_highz.fits"]
plot_P = False
plot_diff = False

kwargs = {"res_label" :  'DESI everest SV1 SNR>4',
          "res_label2" : 'eBOSS DR14 fit',
          "diff_range" : 1.0,
          "no_errors_diff" : True,
          "marker_size" : 5,
          "marker_style" : ".",
          "fonttext" : 12,
          "fontlab" : 11,
          "fontlegend" : 11,
          "z_binsize" : 0.2,
          "kmin" : 4e-2,
          "kmax" : 2.5,
          "grid" : True}


### Noise study mean z


use_diff_noise = False
plot_noise_ratio = True
plot_noise_comparison_mean_k = False
plot_side_band = False
k_units = "A"
fit_asymptote_ratio = True

kwargs_noise1 = {"kmin" : None,
                 "kmax" : None}


### Noise study mean k

plot_noise_comparison_mean_z = False
kwargs_noise2 = {"kmin" : 4e-2,
                 "kmax" : 2.5}


if __name__ == "__main__":
    data = p1d_plot.read_pk_means(pk_means_name)
    p1d_plot.plot_data(data,
                       zbins,
                       outname,
                       plot_P=plot_P,
                       comparison=comparison,
                       comparison_model=comparison_model,
                       comparison_model_file=comparison_model_file,
                       plot_diff=plot_diff,
                       **kwargs)


    p1d_plot.plot_noise_study(data,
                              zbins,
                              outname,
                              k_units,
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
