import numpy as np
import glob, os
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from desitarget.sv1.sv1_targetmask import desi_mask as sv1_desi_mask
from desitarget.sv2.sv2_targetmask import desi_mask as sv2_desi_mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3_desi_mask
from desitarget.targetmask import desi_mask
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras
from p1desi import utils



### DESI spectra extraction



def get_spectra_desi(spectra_path,
                     spectro,
                     cut_objects,
                     compute_diff=False,
                     get_fuji_pb_spectra=False):

    mask_desi_target_names = ["SV1_DESI_TARGET","SV2_DESI_TARGET","SV3_DESI_TARGET","DESI_TARGET"]
    mask_desi_target = [sv1_desi_mask,sv2_desi_mask,sv3_desi_mask,desi_mask]


    flux,pixel_mask,var,target_id,ra,dec = [],[],[],[],[],[]
    if(compute_diff):
        diff_flux=[]
    for path in spectra_path :
        if(spectro == "all"):
            if(compute_diff):
                exp_names = np.sort(glob.glob(os.path.join(path,"*/spectra-*.fits")))
            spectra_names = np.sort(glob.glob(os.path.join(path,"*/coadd-*.fits")))
        else:
            if(compute_diff):
                exp_names = np.sort(glob.glob(os.path.join(path,"*/spectra-{}-*.fits".format(spectro))))
            spectra_names = np.sort(glob.glob(os.path.join(path,"*/coadd-{}-*.fits".format(spectro))))

        for i in range(len(spectra_names)):
            spectra = read_spectra(spectra_names[i])
            if 'brz' not in spectra.bands:
                spectra = coadd_cameras(spectra)
            if 'brz' not in spectra.bands:
                continue
            mask_target = np.full(spectra.fibermap["TARGETID"].shape,False)
            for k in range(len(mask_desi_target_names)):
                if(mask_desi_target_names[k] in spectra.fibermap.colnames):
                    for j in range(len(cut_objects)):
                        mask_target |= ((spectra.fibermap[mask_desi_target_names[k]] & mask_desi_target[k][cut_objects[j]])!=0)
            if("COADD_FIBERSTATUS" in spectra.fibermap.colnames):
                name_fiberstatus = "COADD_FIBERSTATUS"
                mask_target = mask_target & (spectra.fibermap["COADD_FIBERSTATUS"]==0)
            elif("FIBERSTATUS" in spectra.fibermap.colnames):
                name_fiberstatus = "FIBERSTATUS"
            if(get_fuji_pb_spectra):
                mask_target = mask_target & (spectra.fibermap[name_fiberstatus]==0)
            else:
                mask_target = mask_target & ((spectra.fibermap[name_fiberstatus]==0)  |  (spectra.fibermap[name_fiberstatus]==8388608)  |  (spectra.fibermap[name_fiberstatus]==16777216))
            flux.append(spectra.flux['brz'][mask_target])
            pixel_mask.append(spectra.mask['brz'][mask_target])
            ivar=spectra.ivar['brz'][mask_target]
            mask_ivar = (ivar == np.inf) | (ivar <= 10**-8)
            ivar[~mask_ivar] = 1/ivar[~mask_ivar]
            var.append(ivar)
            target_id_to_append = spectra.fibermap["TARGETID"][mask_target]
            target_id.append(target_id_to_append)
            ra.append(spectra.fibermap["TARGET_RA"][mask_target])
            dec.append(spectra.fibermap["TARGET_DEC"][mask_target])
            if(compute_diff):
                exp = read_spectra(exp_names[i])
                for band in ["b","r","z"]:
                    for j in range(len(target_id_to_append)):
                        mask_targetid = exp.fibermap["TARGETID"] == target_id_to_append[j]
                        spectra_exp = exp.flux[band][mask_targetid]
                        ivar_exp = exp.ivar[band][mask_targetid]
                        for k in range(len(spectra_exp)):
                            spectra_exp[k,:] = (-1)**k * spectra_exp[k,:]
                        if(len(spectra_exp)%2 == 1):
                            ivar_exp[-1,:] = np.zeros(ivar_exp[-1,:].shape)
                            spectra_exp[-1,:] = np.zeros(spectra_exp[-1,:].shape)
                        exp.flux[band][mask_targetid] = spectra_exp
                        exp.ivar[band][mask_targetid] = ivar_exp
                exp = coadd_cameras(exp)
                diff_flux.append(exp.flux['brz'][mask_target])


    wavelength = spectra.wave['brz']
    flux = np.concatenate(flux,axis=0)
    pixel_mask = np.concatenate(pixel_mask,axis=0)
    var = np.concatenate(var,axis=0)
    target_id = np.concatenate(target_id,axis=0)
    ra = np.concatenate(ra,axis=0)
    dec = np.concatenate(dec,axis=0)

    mask_pixel = pixel_mask !=0

    flux[mask_pixel] == np.nan
    var[mask_pixel] == np.nan

    if(compute_diff):
        diff_flux=np.concatenate(diff_flux,axis=0)
        diff_flux[mask_pixel] == np.nan
    else:
        diff_flux = None

    return wavelength, flux, diff_flux, var, target_id, ra, dec





### Histo functions


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



### Variance vs Diff fucntions



def plot_variance_histogram(name_out,
                            V_diff,
                            V_pipeline,
                            wavelength,
                            overlap_regions=None,
                            nb_bins=40,
                            outlier_insensitive=False,
                            **kwargs):

    alpha = utils.return_key(kwargs,"alpha",0.3)
    figsize = utils.return_key(kwargs,"figsize",(8,10))
    alpha_fill=utils.return_key(kwargs,"alpha_fill",0.5)
    size_fill=utils.return_key(kwargs,"size_fill",0.2)
    ylim_diff=utils.return_key(kwargs,"ylim_diff",None)
    ylim_ratio=utils.return_key(kwargs,"ylim_ratio",None)

    fig,ax=plt.subplots(3,1,figsize=figsize,sharex=True)
    bin_centers, means1, disp1 = hist_profile(wavelength,V_diff,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(V_diff),np.max(V_diff)),
                                            outlier_insensitive=outlier_insensitive)


    ax[0].errorbar(x=bin_centers, y=means1, yerr=disp1, linestyle='none', marker='.', label="diff",color="b")
    bin_centers, means2, disp2 = hist_profile(wavelength,V_pipeline,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(V_pipeline),np.max(V_pipeline)),
                                            outlier_insensitive=outlier_insensitive)
    ax[0].errorbar(x=bin_centers, y=means2, yerr=disp2, linestyle='none', marker='.', label="ivar",color="r")

    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["diff","pipeline"])

    (ratio_means,ratio_disp) = ( (means1 - means2)/means2,
                                 (means1/means2) * np.sqrt((disp1/means1)**2 + (disp2/means2)**2) )

    (diff_mean,diff_disp) = ( (means1 - means2),
                               np.sqrt((disp1)**2 + (disp2)**2) )

    ratio = (V_diff - V_pipeline)/V_pipeline
    diff = (V_diff - V_pipeline)

    ax[1].plot(wavelength,
               ratio,
               color="r",
               alpha=alpha)

    ax[1].errorbar(x=bin_centers,
                   y=ratio_means,
                   yerr=ratio_disp,
                   linestyle='none',
                   marker='.',
                   color="b")


    bin_centers, means_ratio, disp_ratio = hist_profile(wavelength,ratio,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(ratio),np.max(ratio)),
                                            outlier_insensitive=outlier_insensitive)

    ax[1].errorbar(x=bin_centers,
                   y=means_ratio,
                   yerr=disp_ratio,
                   linestyle='none',
                   marker='.',
                   color="g")


    ax[1].legend(["Ratio","Ratio of bins","Binned ratio"])

    ax[1].set_ylabel("$(Var(diff) - Var(pipeline)) / Var(pipeline)$")

    ax[1].axhline(color="k")
    ax[1].fill_between(wavelength,
                       y1=np.full(wavelength.shape,-size_fill),
                       y2=np.full(wavelength.shape,size_fill),
                       color="gray",
                       alpha=alpha_fill)


    if(ylim_ratio is not None):
        ax[1].set_ylim(ylim_ratio)

    ax[2].plot(wavelength,
               diff,
               color="r",
               alpha=alpha)

    ax[2].errorbar(x=bin_centers,
                   y=diff_mean,
                   yerr=diff_disp,
                   linestyle='none',
                   marker='.',
                   color="b")


    bin_centers, means_diff, disp_diff = hist_profile(wavelength,diff,nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(ratio),np.max(ratio)),
                                            outlier_insensitive=outlier_insensitive)

    ax[2].errorbar(x=bin_centers,
                   y=means_diff,
                   yerr=disp_diff,
                   linestyle='none',
                   marker='.',
                   color="g")


    ax[2].legend(["Difference","Difference of bins","Binned difference"])

    ax[2].set_xlabel("Observed wavelength [$\AA$]")
    ax[2].set_ylabel("$Var(diff) - Var(pipeline)$")

    ax[2].axhline(color="k")

    if(ylim_diff is not None):
        ax[2].set_ylim(ylim_diff)


    if(overlap_regions is not None):
        for i in range(len(overlap_regions)):
            overlap_bin_diff = np.mean(V_diff[(wavelength> overlap_regions[i][0])&(wavelength < overlap_regions[i][1])])
            overlap_bin_pipeline = np.mean(V_pipeline[(wavelength> overlap_regions[i][0])&(wavelength < overlap_regions[i][1])])
            ax[0].plot([np.mean(overlap_regions[i])],[overlap_bin_diff], marker='.',color ="b")
            ax[0].plot([np.mean(overlap_regions[i])],[overlap_bin_pipeline], marker='.',color ="r")
            overlap_bin_ratio = (overlap_bin_diff - overlap_bin_pipeline)/overlap_bin_pipeline
            ax[1].plot([np.mean(overlap_regions[i])],[overlap_bin_ratio], marker='.',color ="b")

            ax[0].fill_between(wavelength,np.min(means1), np.max(means1),
                               where= ((wavelength> overlap_regions[0][0])&
                                       (wavelength < overlap_regions[0][1]))|
                                       ((wavelength> overlap_regions[1][0])&
                                       (wavelength < overlap_regions[1][1])),
                               color='green',
                               alpha=0.25)
            ax[1].fill_between(wavelength,np.min(means1), np.max(means1),
                               where= ((wavelength> overlap_regions[0][0])&
                                       (wavelength < overlap_regions[0][1]))|
                                       ((wavelength> overlap_regions[1][0])&
                                       (wavelength < overlap_regions[1][1])),
                               color='green',
                               alpha=0.25)



    fig.savefig(f"{name_out}.pdf",format="pdf")
    plt.close()





def plot_variance(name_out,V_diff,V_pipeline,wavelength,overlap_regions=None,points=False,**kwargs):

    figsize = utils.return_key(kwargs,"figsize",(8,10))
    alpha_fill=utils.return_key(kwargs,"alpha_fill",0.5)
    size_fill=utils.return_key(kwargs,"size_fill",0.2)


    if(points):
        linestyle='none'
        marker='.'
    else:
        linestyle=None
        marker=None
    ratio = (V_diff - V_pipeline)/V_pipeline
    diff = (V_diff - V_pipeline)
    fig,ax=plt.subplots(3,1,figsize=figsize,sharex=True)

    ax[0].plot(wavelength,V_diff, label="diff",color="b",linestyle=linestyle,marker=marker)
    ax[0].plot(wavelength,V_pipeline, label="pipeline",color="r",linestyle=linestyle,marker=marker)
    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["diff","pipeline"])




    ax[1].plot(wavelength,ratio, label="ratio",color="b",linestyle=linestyle,marker=marker)
    ax[1].set_ylabel("$(Var(diff) - Var(pipeline)) / Var(pipeline)$")
    ax[1].axhline(color="k")
    ax[1].fill_between(wavelength,
                       y1=np.full(wavelength.shape,-size_fill),
                       y2=np.full(wavelength.shape,size_fill),
                       color="gray",
                       alpha=alpha_fill)

    ax[2].plot(wavelength,diff, label="diff",color="b",linestyle=linestyle,marker=marker)
    ax[2].set_xlabel("Observed wavelength [$\AA$]")
    ax[2].set_ylabel("$Var(diff) - Var(pipeline)$")
    ax[2].axhline(color="k")



    if(overlap_regions is not None):
        ax[0].fill_between(wavelength,min(np.min(V_diff),np.min(V_pipeline)), max(np.max(V_diff),np.max(V_pipeline)), where= ((wavelength> overlap_regions[0][0])&(wavelength < overlap_regions[0][1]))|((wavelength> overlap_regions[1][0])&(wavelength < overlap_regions[1][1])),color='green', alpha=0.25)
        ax[1].fill_between(wavelength,np.min(ratio), np.max(ratio), where= ((wavelength> overlap_regions[0][0])&(wavelength < overlap_regions[0][1]))|((wavelength> overlap_regions[1][0])&(wavelength < overlap_regions[1][1])),color='green', alpha=0.25)

    fig.savefig(f"{name_out}.pdf",format="pdf")
    plt.close()



### Flux Variance vs Noise functions

def plot_hist(name_out,
              var_pipeline,
              diff,
              wavelength,
              nb_bins,
              **kwargs):

    V_diff = np.nanvar(diff,axis=0)
    V_pipeline = np.nanmean(var_pipeline,axis=0)

    plot_variance(f"{name_out}_raw",
                  V_diff,
                  V_pipeline,
                  wavelength,
                  **kwargs)

    plot_variance_histogram(f"{name_out}_histogram",
                            V_diff,
                            V_pipeline,
                            wavelength,
                            nb_bins=nb_bins,
                            outlier_insensitive=False,
                            **kwargs)

    plot_variance_histogram(f"{name_out}_histogram_bin_outlier_insensitive",
                            V_diff,
                            V_pipeline,
                            wavelength,
                            nb_bins=nb_bins,
                            outlier_insensitive=True,
                            **kwargs)



def plot_hist_var_outlier_insensitive(name_out,
                                      var_pipeline,
                                      diff,
                                      wavelength,
                                      nb_bins,
                                      **kwargs):

    V_diff_outliers = ((np.nanpercentile(diff,84.135,axis=0)-np.nanpercentile(diff,15.865,axis=0))/2)**2
    V_pipeline_outliers = np.nanmedian(var_pipeline,axis=0)

    plot_variance(f"{name_out}_raw_var_outlier_insensitive",
                  V_diff_outliers,
                  V_pipeline_outliers,
                  wavelength,
                  **kwargs)


    plot_variance_histogram(f"{name_out}_histogram_var_outlier_insensitive",
                            V_diff_outliers,
                            V_pipeline_outliers,
                            wavelength,
                            nb_bins=nb_bins,
                            outlier_insensitive=False,
                            **kwargs)

    plot_variance_histogram(f"{name_out}_histogram_var_outlier_insensitive_bin_outlier_insensitive",
                            V_diff_outliers,
                            V_pipeline_outliers,
                            wavelength,
                            nb_bins=nb_bins,
                            outlier_insensitive=True,
                            **kwargs)



def plot_2d_hist(name_out,
                 var_pipeline,
                 diff,
                 wavelength,
                 nb_bins,
                 **kwargs):

    bin_wave,V_pipeline = hist_profile_2d_bins(wavelength,
                                               var_pipeline,
                                               nb_bins,
                                               statistic="mean",
                                               outlier_insensitive=False)

    bin_wave,V_diff = hist_profile_2d_bins(wavelength,
                                           diff,
                                           nb_bins,
                                           statistic="var",
                                           outlier_insensitive=False)

    plot_variance(f"{name_out}_2d_histogram",
                  V_diff,
                  V_pipeline,
                  bin_wave,
                  points=True,
                  **kwargs)


    bin_wave,V_pipeline = hist_profile_2d_bins(wavelength,
                                               var_pipeline,
                                               nb_bins,
                                               statistic="mean",
                                               outlier_insensitive=True)

    bin_wave,V_diff = hist_profile_2d_bins(wavelength,
                                           diff,
                                           nb_bins,
                                           statistic="var",
                                           outlier_insensitive=True)

    plot_variance(f"{name_out}_2d_histogram_outlier_insensitive",
                  V_diff,
                  V_pipeline,
                  bin_wave,
                  points=True,
                  **kwargs)






#### Other plot functions Obsolete



def plot_noise_fluxvariance_comparison(V_coadd,V_pipeline,wavelength,name_out,nb_bins=40):
    fig,ax=plt.subplots(2,1,figsize=(8,5),sharex=True)
    ax[0].plot(wavelength,V_coadd, label="diff",color="b")
    ax[0].plot(wavelength,V_pipeline, label="pipeline",color="r")
    ax[0].set_ylabel("$Var_i$")
    ax[0].legend(["coadd","pipeline"])


    bin_centers, means, disp = hist_profile(wavelength,
                                            V_coadd,
                                            nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(V_coadd),np.max(V_coadd)))
    ax[1].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="coadd",color="b")
    bin_centers, means, disp = hist_profile(wavelength,
                                            V_pipeline,
                                            nb_bins,
                                            (np.min(wavelength),np.max(wavelength)),
                                            (np.min(V_pipeline),np.max(V_pipeline)))
    ax[1].errorbar(x=bin_centers, y=means, yerr=disp, linestyle='none', marker='.', label="pipeline",color="r")
    ax[1].set_ylabel("$Var_i$ rebin")


    fig.savefig(f"{name_out}_noise_fluxvariance_comparison.pdf",format="pdf")





def plot_noise_fluxvariance_ratio(name_out,spectra_array,noise_array,wavelength,nb_bins,outlier_insensitive=False):
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

    fig.savefig(f"{name_out}_noise_fluxvariance_ratio.pdf",format="pdf")




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
