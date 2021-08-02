import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from desispec.io import read_sky
from scipy.ndimage.filters import gaussian_filter


lambdaLy = 1215.673123130217

def plot_skylines_on_sky_fiber(file_name,
                              line_file,
                              name_out,
                              ylim=[0,500],
                              median_size=40):
    fig,ax=plt.subplots(2,1,figsize=(8,8),sharex=True)

    for i in range(len(file_name)):
        sky=read_sky(file_name[i])
        median_sky = np.mean(np.array([median_filter(sky.flux[j],median_size) for j in range(len(sky.flux))]),axis=0)
        mean_flux = np.mean(sky.flux,axis=0)
        ax[0].plot(sky.wave,mean_flux)
        ax[1].plot(sky.wave,median_sky)

    ax[0].legend(["B","R","Z"])

    line_to_plot = np.loadtxt(line_file, usecols=range(1,3))
    for i in range(len(line_to_plot)):
        x = np.linspace(line_to_plot[i][0],line_to_plot[i][1],100)
        for j in [0,1]:
            ax[j].fill_between(x,
                             ylim[0],
                             ylim[1],
                             color = f"C{1}",
                             alpha=0.3)

    ax[0].set_ylabel("Mean Sky fiber flux")
    ax[1].set_ylabel("Mean of Sky fiber flux median filtered")

    ax[0].set_ylim([ylim[0],ylim[1]])
    ax[1].set_ylim([ylim[0],ylim[1]])
    plt.savefig(f"{name_out}_sky_fiber.pdf",format="pdf")


def compute_line_width(line,
                       method_width,
                       width_angstrom=1,
                       threshold_width=1.2,
                       mean_flux=None,
                       median_sky=None,
                       wave=None):

    if(method_width.lower() == "constant"):
        line[0] = line[0] - width_angstrom
        line[1] = line[1] + width_angstrom

    elif(method_width.lower() == "threshold"):
        mask_width = mean_flux < threshold_width*median_sky
        if(len(wave[mask_width&(wave<=line[0])]) == 0):
            line[0] = line[0] - width_angstrom
        else:
            line[0] = np.max(wave[mask_width&(wave<=line[0])]) - 0.1
        if(len(wave[mask_width&(wave>=line[1])]) == 0):
            line[1] = line[1] + width_angstrom
        else:
            line[1] = np.min(wave[mask_width&(wave>=line[1])]) + 0.1
    return(line)


def get_lines_from_sky_file(file_name,
                            threshold,
                            method_width,
                            median_size,
                            threshold_width=[1.2,1.2,1.2],
                            width_angstrom=1):
    lines = []
    for j in range(len(file_name)):
        sky=read_sky(file_name[j])
        median_sky = np.mean(np.array([median_filter(sky.flux[j],median_size) for j in range(len(sky.flux))]),axis=0)
        mean_flux = np.mean(sky.flux,axis=0)
        mask = mean_flux > threshold[j]*median_sky
        wavelength = sky.wave[mask]
        diff = wavelength[1:] - wavelength[:-1]
        arg_peak = np.argwhere(diff > 1)
        if(len(arg_peak) == 0):
            if(len(wavelength)!=0):
                line = compute_line_width([np.min(wavelength),np.max(wavelength)],
                                          method_width,
                                          width_angstrom=width_angstrom,
                                          threshold_width=threshold_width[j],
                                          mean_flux=mean_flux,
                                          median_sky=median_sky,
                                          wave=sky.wave)
                lines.append(line)
        else:
            arg_init_peak = 0
            for i in range(len(arg_peak)):
                line = compute_line_width([wavelength[arg_init_peak],wavelength[arg_peak[i][0]]],
                                          method_width,
                                          width_angstrom=width_angstrom,
                                          threshold_width=threshold_width[j],
                                          mean_flux=mean_flux,
                                          median_sky=median_sky,
                                          wave=sky.wave)
                lines.append(line)
                arg_init_peak = arg_peak[i][0] + 1
            line = compute_line_width([wavelength[arg_peak[-1][0]+1],wavelength[-1]],
                                      method_width,
                                      width_angstrom=width_angstrom,
                                      threshold_width=threshold_width[j],
                                      mean_flux=mean_flux,
                                      median_sky=median_sky,
                                      wave=sky.wave)
            lines.append(line)
    return(np.array(lines))


def add_custom_line(lines,custom_lines,names,types):
    list_type = []
    list_name = []
    lines_out = []
    custom_lines_mask = np.full(len(custom_lines),True)
    for i in range(len(lines)):
        custom_lines_left = np.argwhere(custom_lines_mask)
        for arg in custom_lines_left:
            if(custom_lines[arg[0]][1] < lines[i][0]):
                list_type.append(custom_lines[arg[0]][3])
                list_name.append(custom_lines[arg[0]][0])
                lines_out.append([custom_lines[arg[0]][1],custom_lines[arg[0]][2]])
                custom_lines_mask[arg[0]] = False
        list_type.append(types)
        list_name.append(names)
        lines_out.append(lines[i])
    return(lines_out,list_name,list_type)

def merge_overlaping_line(lines):
    lines = np.array(sorted(lines, key=lambda a : a[0], reverse=False))
    starts = lines[:,0]
    ends = np.maximum.accumulate(lines[:,1])
    valid = np.zeros(len(lines) + 1, dtype=np.bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = starts[1:] >= ends[:-1]
    return np.vstack((starts[:][valid[:-1]], ends[:][valid[1:]])).T



def delete_line_number(i,names,lines,types):
    del names[i]
    del lines[i]
    del types[i]


def replace_name(center_line,new_name,names,lines,types):
    for i in range(len(lines)):
        if((lines[i][0] < center_line)&(lines[i][1] > center_line)):
            names[i] = new_name
    return(names,lines,types)

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
                          lines,
                          outlier_insensitive=False,
                          gaussian_smoothing=40):
    if(outlier_insensitive):
        mean_spectra = np.nanmedian(flux,axis=0)
    else:
        mean_spectra = np.nanmean(flux,axis=0)

    fig,ax=plt.subplots(4,1,figsize=(8,8),sharex=True)
    ax[0].plot(wavelength,mean_spectra)
    ax[0].set_ylabel("Mean spectra")


    # mean_median =np.mean(np.array([median_filter(flux[i],median_smoothing) for i in range(len(flux))]),axis=0)
    # mean_median =median_filter(mean_spectra,median_smoothing)

    gaussian_mean =gaussian_filter(mean_spectra,gaussian_smoothing)


    ax[1].plot(wavelength,gaussian_mean,color="b")
    ax[1].set_ylabel("Mean spectra smoothed")


    ratio = np.abs((gaussian_mean - mean_spectra)/gaussian_mean)
    ax[2].plot(wavelength,ratio,color="b")
    ax[3].plot(wavelength,ratio,color="b")
    ax[2].set_ylabel("Ratio + eBOSS lines")
    ax[3].set_ylabel("Ratio + DESI lines")
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
                               alpha=0.3)


    fig.savefig(f"{name_out}_mean_spectra.pdf",format="pdf")


def plot_centered_lines(name_out,
                        array_list,
                        wavelength,
                        lines,
                        outlier_insensitive=False,
                        diff_lambda=10,
                        title="Mean flux"):
    if(outlier_insensitive):
        mean_spectra = np.nanmedian(array_list,axis=0)
    else:
        mean_spectra = np.nanmean(array_list,axis=0)


    line_to_plot = np.loadtxt(lines, usecols=range(1,3))
    for i in range(len(line_to_plot)):

        fig,ax=plt.subplots(2,2,figsize=(8,8),gridspec_kw={'height_ratios': [3, 1],'width_ratios': [3, 1]},sharex=False)
        mean_line = (line_to_plot[i][0] + line_to_plot[i][1]) / 2

        mask = wavelength >= mean_line - diff_lambda
        mask &= wavelength <= mean_line + diff_lambda

        ax[0][0].plot(wavelength[mask],mean_spectra[mask],"x-")
        ax[0][0].grid()
        ax[0][0].set_title(title)
        x = np.linspace(line_to_plot[i][0],line_to_plot[i][1],100)
        ax[0][0].fill_between(x,
                              np.min(mean_spectra[mask]),
                              np.max(mean_spectra[mask]),
                              color = f"C{1}",
                              alpha=0.3)

        ax[0][1].plot(wavelength,mean_spectra)
        x = np.linspace(line_to_plot[i][0],line_to_plot[i][1],100)
        ax[0][1].fill_between(x,
                              np.min(mean_spectra),
                              np.max(mean_spectra),
                              color = f"C{1}",
                              alpha=0.3)
        ax[0][1].set_xlim([mean_line - diff_lambda,mean_line + diff_lambda])


        ax[1][0].plot(wavelength,mean_spectra)
        ax[1][0].fill_between(x,
                              np.min(mean_spectra),
                              np.max(mean_spectra),
                              color = f"C{1}",
                              alpha=0.3)
        ax[1][0].set_xlabel("Observed wavelength")
        fig.delaxes(ax[1][1])
        fig.savefig(f"{name_out}_line_centered_{i:03d}_{np.around(mean_line,0)}.png",format="png")
        plt.close()


def compute_length_masked(lines,redshift_bins):
    percentage_mask = []
    line_to_plot = np.loadtxt(lines, usecols=range(1,3))
    for i in range(len(redshift_bins)):
        lambda_min = (1 + redshift_bins[i] - 0.1)* lambdaLy
        lambda_max = (1 + redshift_bins[i] + 0.1)* lambdaLy
        sum = 0
        for j in range(len(line_to_plot)):
            if((line_to_plot[j][1] < lambda_max)&(line_to_plot[j][0] > lambda_min)):
                sum =sum +  line_to_plot[j][1] - line_to_plot[j][0]
        percentage_mask.append(100* sum / (lambda_max - lambda_min))
    return(percentage_mask)

def plot_percentage_mask(redshift_bins,percentage_mask,legend,nameout):
    plt.figure()
    plt.ylabel("Percentage masked")
    for i in range(len(percentage_mask)):
        plt.plot(redshift_bins,percentage_mask[i])
    plt.legend(legend)
    plt.savefig(f"{nameout}.pdf",format ="pdf")
