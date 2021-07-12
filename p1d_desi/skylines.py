import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from desispec.io import read_sky




def get_lines_from_sky_file(file_name,threshold,width_pix):
    width_ang = width_pix * 0.8
    lines = []
    for file in file_name:
        sky=read_sky(file)
        mean_flux = np.mean(sky.flux,axis=0)
        mask = mean_flux > threshold*np.median(mean_flux)
        wavelength = sky.wave[mask]
        diff = wavelength[1:] - wavelength[:-1]
        arg_peak = np.argwhere(diff > 1)
        if(len(arg_peak) != 0):
            arg_init_peak = 0
            for i in range(len(arg_peak)):
                lines.append([wavelength[arg_init_peak] - width_ang/2,wavelength[arg_peak[i][0]]+ width_ang/2])
                arg_init_peak = arg_peak[i][0] + 1
            lines.append([wavelength[arg_peak[-1][0]+1] - width_ang/2,wavelength[-1] + width_ang/2])
    return(lines)


def write_skyline_file(name_out,names,lines,types):
    if type(names)!=list:
        names = [names for i in range(len(lines))]
    if type(types)!=list:
        types = [types for i in range(len(lines))]
    file = open(name_out,"w")
    header = """#
# File to veto lines either in observed or rest frame
#       or even rest frame of DLA
#
#  Veto lines used for Pk1D analysis (DESI SV)
#
# lambda is given in Angstrom
#
# name  lambda_min  lambda_max  ('OBS' or 'RF' or 'RF_DLA')
#\n"""
    file.write(header)
    for i in range(len(lines)):
        file.write(f"{names[i]}  {np.round(lines[i][0],3)}  {np.round(lines[i][1],3)}  {types[i]}\n")
    file.close()




def plot_skyline_analysis(name_out,
                          flux,
                          wavelength,
                          nb_bins,
                          lines,
                          out_line,
                          outlier_insensitive=False,
                          gaussian_smoothing=2):
    if(outlier_insensitive):
        mean_spectra = np.nanmedian(flux,axis=0)
    else:
        mean_spectra = np.nanmean(flux,axis=0)

    fig,ax=plt.subplots(4,1,figsize=(8,8),sharex=True)
    ax[0].plot(wavelength,mean_spectra)
    ax[0].set_ylabel("Mean spectra")


    smoothed_mean = gaussian_filter(mean_spectra,gaussian_smoothing)


    ax[1].plot(wavelength,smoothed_mean,color="b")
    ax[1].set_ylabel("Mean spectra smoothed")


    ratio = np.abs((smoothed_mean - mean_spectra)/smoothed_mean)
    ax[2].plot(wavelength,ratio,color="b")
    ax[3].plot(wavelength,ratio,color="b")
    ax[2].set_ylabel("ratio smoothed not smoothed")
    j = 1
    for line in lines:
        j = j +1
        line_to_plot = np.loadtxt(line, usecols=range(1,3))
        for i in range(len(line_to_plot)):
            ax[j].fill_between(wavelength,
                               np.min(ratio),
                               np.max(ratio),
                               where= ((wavelength> line_to_plot[i][0])&(wavelength < line_to_plot[i][1])),
                               color = f"C{1}",
                               alpha=0.5)
    ax[2].legend(["mean","eBOSS","DESI sky"])







    fig.savefig(f"{name_out}_mean_spectra.pdf",format="pdf")
