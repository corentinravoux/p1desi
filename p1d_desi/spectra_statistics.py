import numpy as np
import fitsio
import glob, os
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from desitarget.sv1.sv1_targetmask import desi_mask as sv1_desi_mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3_desi_mask
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras
from scipy.ndimage.filters import gaussian_filter



def hist_profile(x, y, bins, range_x,range_y,outlier_insensitive=False):
    w = (y>range_y[0]) & (y<range_y[1])
    if(outlier_insensitive):
        means_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic='median')
        outlier_insensitive_std = lambda x : (np.nanpercentile(x,84.135,axis=0)-np.nanpercentile(x,15.865,axis=0))/2
        std_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic=outlier_insensitive_std)
        nb_entries_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic='count')

    else:
        means_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic='mean')
        std_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic='std')
        nb_entries_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic='count')


    means = means_result.statistic
    std = std_result.statistic
    nb_entries = nb_entries_result.statistic

    errors = std/np.sqrt(nb_entries)

    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return(bin_centers, means, errors)

def var_profile(x, y, bins, range_x,range_y,outlier_insensitive=False):
    w = (y>range_y[0]) & (y<range_y[1])
    if(outlier_insensitive):
        outlier_insensitive_std = lambda x : (np.nanpercentile(x,84.135,axis=0)-np.nanpercentile(x,15.865,axis=0))/2
        std_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic=outlier_insensitive_std)
    else:
        std_result = binned_statistic(x[w], y[w], bins=bins, range=range_x, statistic='std')

    std = std_result.statistic

    bin_edges = std_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return(bin_centers, std)



def hist_profile_2d_bins(x, y, bins,statistic="mean",outlier_insensitive=False):

    bin_edges = np.array([i*((x[-1] -x[0])/bins) + x[0] for i in range(bins)] + [x[-1]])

    bin_2d_stat = []
    for i in range(len(bin_edges)-1):
        mask = (x >= bin_edges[i])&(x < bin_edges[i+1])
        if(i == len(bin_edges) -2) :
            mask |=(x == bin_edges[i+1])
        bin_y = np.transpose(y)[mask]
        if(outlier_insensitive):
            if(statistic == "mean"):
                bin_2d_stat.append(np.nanmedian(bin_y))
            elif(statistic == "var"):
                outlier_insensitive_std = lambda x : (np.nanpercentile(x,84.135)-np.nanpercentile(x,15.865))/2
                bin_2d_stat.append(outlier_insensitive_std(bin_y)**2)
        else:
            if(statistic == "mean"):
                bin_2d_stat.append(np.nanmean(bin_y))
            elif(statistic == "var"):
                bin_2d_stat.append(np.nanstd(bin_y)**2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return(bin_centers, np.array(bin_2d_stat))




def plot_variance_histogram(name_out,V_diff,V_pipeline,wavelength,overlap_regions=None,nb_bins=40,outlier_insensitive=False):
    fig,ax=plt.subplots(2,1,figsize=(8,5),sharex=True)
    bin_centers, means, disp = hist_profile(wavelength,V_diff,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(V_diff),np.max(V_diff)),
                                            outlier_insensitive=outlier_insensitive)
    ax[0].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="diff",color="b")
    means_ratio = means
    disp_ratio = disp
    bin_centers, means, disp = hist_profile(wavelength,V_pipeline,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(V_pipeline),np.max(V_pipeline)),
                                            outlier_insensitive=outlier_insensitive)
    ax[0].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="ivar",color="r")

    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["diff","pipeline"])

    (means_ratio,disp_ratio) = ( (means_ratio - means)/means,
                                 (means_ratio/means) * np.sqrt((disp_ratio/means_ratio)**2 + (disp/means)**2) )

    ratio = (V_diff - V_pipeline)/V_pipeline

    ax[1].plot(wavelength,ratio, label="raw",color="r",alpha=0.3)
    ax[1].errorbar(x=bin_centers, y=means_ratio, yerr=disp_ratio, linestyle='none', marker='.', label="binned",color="b")


    bin_centers, means, disp = hist_profile(wavelength,ratio,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(ratio),np.max(ratio)),
                                            outlier_insensitive=outlier_insensitive)
    ax[1].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="ivar",color="g")


    ax[1].legend(["raw","binned"])

    ax[1].set_xlabel("Observed wavelength [$\AA$]")
    ax[1].set_ylabel("$(Var(diff) - Var(pipeline)) / Var(pipeline)$")

    if(overlap_regions is not None):
        for i in range(len(overlap_regions)):
            overlap_bin_diff = np.mean(V_diff[(wavelength> overlap_regions[i][0])&(wavelength < overlap_regions[i][1])])
            overlap_bin_pipeline = np.mean(V_pipeline[(wavelength> overlap_regions[i][0])&(wavelength < overlap_regions[i][1])])
            ax[0].plot([np.mean(overlap_regions[i])],[overlap_bin_diff], marker='.',color ="b")
            ax[0].plot([np.mean(overlap_regions[i])],[overlap_bin_pipeline], marker='.',color ="r")
            overlap_bin_ratio = (overlap_bin_diff - overlap_bin_pipeline)/overlap_bin_pipeline
            ax[1].plot([np.mean(overlap_regions[i])],[overlap_bin_ratio], marker='.',color ="b")

            ax[0].fill_between(wavelength,np.min(means), np.max(means),
                               where= ((wavelength> overlap_regions[0][0])&
                                       (wavelength < overlap_regions[0][1]))|
                                       ((wavelength> overlap_regions[1][0])&
                                       (wavelength < overlap_regions[1][1])),
                               color='green',
                               alpha=0.25)
            ax[1].fill_between(wavelength,np.min(means), np.max(means),
                               where= ((wavelength> overlap_regions[0][0])&
                                       (wavelength < overlap_regions[0][1]))|
                                       ((wavelength> overlap_regions[1][0])&
                                       (wavelength < overlap_regions[1][1])),
                               color='green',
                               alpha=0.25)



    fig.savefig("{}_hist.pdf".format(name_out),format="pdf")
    plt.close()





def plot_variance(name_out,V_diff,V_pipeline,wavelength,overlap_regions=None,points=False):
    if(points):
        linestyle='none'
        marker='.'
    else:
        linestyle=None
        marker=None
    ratio = (V_diff - V_pipeline)/V_pipeline
    fig,ax=plt.subplots(2,1,figsize=(8,5),sharex=True)

    ax[0].plot(wavelength,V_diff, label="diff",color="b",linestyle=linestyle,marker=marker)
    ax[0].plot(wavelength,V_pipeline, label="pipeline",color="r",linestyle=linestyle,marker=marker)
    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["diff","pipeline"])

    ax[1].plot(wavelength,ratio, label="ratio",color="b",linestyle=linestyle,marker=marker)
    ax[1].set_xlabel("Observed wavelength [$\AA$]")
    ax[1].set_ylabel("$(Var(diff) - Var(pipeline)) / Var(pipeline)$")

    if(overlap_regions is not None):
        ax[0].fill_between(wavelength,min(np.min(V_diff),np.min(V_pipeline)), max(np.max(V_diff),np.max(V_pipeline)), where= ((wavelength> overlap_regions[0][0])&(wavelength < overlap_regions[0][1]))|((wavelength> overlap_regions[1][0])&(wavelength < overlap_regions[1][1])),color='green', alpha=0.25)
        ax[1].fill_between(wavelength,np.min(ratio), np.max(ratio), where= ((wavelength> overlap_regions[0][0])&(wavelength < overlap_regions[0][1]))|((wavelength> overlap_regions[1][0])&(wavelength < overlap_regions[1][1])),color='green', alpha=0.25)

    fig.savefig("{}.pdf".format(name_out),format="pdf")
    plt.close()



def plot_spectra_ratio(V_coadd,V_pipeline,wavelength,name_out,nb_bins=40):
    fig,ax=plt.subplots(2,1,figsize=(8,5),sharex=True)
    ax[0].plot(wavelength,V_coadd, label="diff",color="b")
    ax[0].plot(wavelength,V_pipeline, label="pipeline",color="r")
    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["coadd","pipeline"])
    bin_centers, means, disp = hist_profile(wavelength,V_coadd,nb_bins,(np.min(wavelength),np.max(wavelength)),(np.min(V_coadd),np.max(V_coadd)))
    ax[1].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="coadd",color="b")
    bin_centers, means, disp = hist_profile(wavelength,V_pipeline,nb_bins,(np.min(wavelength),np.max(wavelength)),(np.min(V_pipeline),np.max(V_pipeline)))
    ax[1].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="pipeline",color="r")


    fig.savefig(f"{name_out}_spectra_ratio.pdf",format="pdf")


def plot_ra_dec_diagram(name_out,ra,dec,cut_objects):
    plt.figure()
    for i in range(len(cut_objects)):
        ra[i] = np.concatenate(ra[i],axis=0)
        dec[i] = np.concatenate(dec[i],axis=0)
        plt.plot(ra[i],dec[i],"+")
    plt.grid()
    plt.legend(["Target {}".format(obj)for obj in cut_objects])
    plt.xlabel("RA(J2000) [deg]")
    plt.ylabel("DEC(J2000) [deg]")
    plt.savefig(f"radec_diagram_{name_out}.pdf",format="pdf")
    plt.close()


def plot_skyline_analysis(name_out,
                          flux,
                          wavelength,
                          nb_bins,
                          lines_eBOSS,
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

    lines_old = np.loadtxt(lines_eBOSS, usecols=range(1,3))

    ratio = np.abs((smoothed_mean - mean_spectra)/smoothed_mean)
    ax[2].plot(wavelength,ratio,color="b")
    ax[2].set_ylabel("ratio smoothed not smoothed")

    for i in range(len(lines_old)):
        ax[2].fill_between(wavelength,
                           np.min(ratio),
                           np.max(ratio),
                           where= ((wavelength> lines_old[i][0])&(wavelength < lines_old[i][1])),
                           color='red',
                           alpha=0.2)

    ax[3].plot(wavelength,np.gradient(ratio),color="b")

    fig.savefig(f"{name_out}_mean_spectra.pdf",format="pdf")



def plot_spectra_noise_comparison(name_out,spectra_array,noise_array,wavelength,nb_bins,outlier_insensitive=False):
    mean_noise = []
    var_flux = []
    for i in range(len(spectra_array)):
        bin_centers, means, errors = hist_profile(wavelength,
                                                  noise_array[i],
                                                  nb_bins,
                                                  (np.min(wavelength),np.max(wavelength)),
                                                  (np.min(noise_array[i]),np.max(noise_array[i])),
                                                  outlier_insensitive=outlier_insensitive)
        mean_noise.append(means)
        bin_centers, std = var_profile(wavelength,
                                       spectra_array[i],
                                       nb_bins,
                                       (np.min(wavelength),np.max(wavelength)),
                                       (np.min(spectra_array[i]),np.max(spectra_array[i])),
                                       outlier_insensitive=outlier_insensitive)
        var_flux.append(std)

    V_coadd = np.nanmean(var_flux,axis=0)
    V_pipeline = np.nanmean(mean_noise,axis=0)
    fig,ax=plt.subplots(2,1,figsize=(8,5),sharex=True)
    ax[0].plot(bin_centers,V_coadd, label="diff",color="b", linestyle='none', marker='.',)
    ax[0].plot(bin_centers,V_pipeline, label="pipeline",color="r", linestyle='none', marker='.',)
    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["coadd","pipeline"])

    ax[1].set_ylabel("$(Var(flux) - Mean(var)) / Mean(var)$")

    ax[1].plot(bin_centers,(V_coadd-V_pipeline)/V_pipeline, label="diff",color="k", linestyle='none', marker='.',)

    fig.savefig(f"{name_out}_spectra_noise_ratio.pdf",format="pdf")


def get_spectra_desi(spectra_path,
                     spectro,
                     cut_objects,
                     survey,
                     diff=False):


    if(survey.upper() == "SV1"):
        desi_mask_used = sv1_desi_mask
        target_key = "SV1_DESI_TARGET"
    if(survey.upper() == "SV3"):
        desi_mask_used = sv3_desi_mask
        target_key = "SV3_DESI_TARGET"



    flux,pixel_mask,var,target_class,target_id,ra,dec = [],[],[],[],[],[],[]
    for path in spectra_path :
        if(diff):
            diff_flux=[]

        if(spectro == "all"):
            spectra_names = np.sort(glob.glob(os.path.join(path,"*/coadd-*.fits")))
        else:
            spectra_names = np.sort(glob.glob(os.path.join(path,"*/coadd-{}-*.fits".format(spectro))))


        for i in range(len(spectra_names)):
            spectra = read_spectra(spectra_names[i])
            if 'brz' not in spectra.bands:
                spectra = coadd_cameras(spectra)
            mask_target = np.full(spectra.fibermap[target_key].shape,False)
            for j in range(len(cut_objects)):
                mask_target |= ((spectra.fibermap[target_key] & desi_mask_used[cut_objects[j]])!=0)
            mask_target = mask_target & (spectra.fibermap["FIBERSTATUS"]==0)

            flux.append(spectra.flux['brz'][mask_target])
            pixel_mask.append(spectra.mask['brz'][mask_target])
            ivar=spectra.ivar['brz'][mask_target]
            mask_ivar = (ivar == np.inf) | (ivar <= 10**-8)
            ivar[~mask_ivar] = 1/ivar[~mask_ivar]
            var.append(ivar)
            target_id.append(spectra.fibermap["TARGETID"][mask_target])
            target_class.append(spectra.fibermap[target_key][mask_target])
            ra.append(spectra.fibermap["TARGET_RA"][mask_target])
            dec.append(spectra.fibermap["TARGET_DEC"][mask_target])
            if(diff):
                diff_flux.append(spectra.extra['brz']["DIFF_FLUX"][mask_target])


    wavelength = spectra.wave['brz']
    flux = np.concatenate(flux,axis=0)
    pixel_mask = np.concatenate(pixel_mask,axis=0)
    var = np.concatenate(var,axis=0)
    target_class = np.concatenate(target_class,axis=0)
    target_id = np.concatenate(target_id,axis=0)
    ra = np.concatenate(ra,axis=0)
    dec = np.concatenate(dec,axis=0)

    mask_pixel = pixel_mask !=0

    flux[mask_pixel] == np.nan
    var[mask_pixel] == np.nan

    if(diff):
        diff_flux=np.concatenate(diff_flux,axis=0)
        diff_flux[mask_pixel] == np.nan


    return wavelength, flux, var, target_class, target_id, ra, dec
