import numpy as np
import fitsio
import glob, os
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from desitarget.sv1.sv1_targetmask import desi_mask as sv1_desi_mask



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



def get_spectra(spectra_path,spectro,cut_objects,target_key):
        if(spectro == "all"):
            spectra_names = np.sort(glob.glob(os.path.join(spectra_path,"coadd-*.fits")))
        else:
            spectra_names = np.sort(glob.glob(os.path.join(spectra_path,"coadd-{}-*.fits".format(spectro))))

        spectra,diff_spectra,target_id,var,wavelength = [],[],[],[],[]
        pixel_mask = []
        ra,dec = [[]for i in range(len(cut_objects))],[[]for i in range(len(cut_objects))]
        masktarget = []
        for i in range(len(spectra_names)):
            h = fitsio.FITS(spectra_names[i])
            target = h["FIBERMAP"][target_key][:]
            mask = np.full(target.shape,False)
            fiberstatus_sp = h["FIBERMAP"]["FIBERSTATUS"][:]
            for i in range(len(cut_objects)):
                mask |= ((target & sv1_desi_mask[cut_objects[i]])!=0)
                masktarget.append(((target & sv1_desi_mask[cut_objects[i]])!=0))
            mask = mask & (fiberstatus_sp==0)
            for i in range(len(cut_objects)):
                masktarget[i] &= (fiberstatus_sp==0)
            spectra.append(h["BRZ_FLUX"][:,:][mask])
            diff_spectra.append(h["BRZ_DIFF_FLUX"][:,:][mask])
            pixel_mask.append(h["BRZ_MASK"][:,:][mask])
            wavelength.append(h["BRZ_WAVELENGTH"][:])
            ivar_to_append = h["BRZ_IVAR"][:,:]
            mask_ivar = (ivar_to_append == np.inf) | (ivar_to_append <= 10**-8)
            ivar_to_append[~mask_ivar] = 1/ivar_to_append[~mask_ivar]
            ivar_to_append[mask_ivar] = np.nan
            var.append(ivar_to_append[mask])
            target_id.append(h["FIBERMAP"]["TARGETID"][:][mask])
            for i in range(len(cut_objects)):
                ra[i].append(h["FIBERMAP"]["TARGET_RA"][:][masktarget[i]])
                dec[i].append(h["FIBERMAP"]["TARGET_DEC"][:][masktarget[i]])


        wavelength = np.nanmean(wavelength,axis=0)
        spectra = np.concatenate(spectra,axis=0)
        diff = np.concatenate(diff_spectra,axis=0)
        var = np.concatenate(var,axis=0)
        target_id = np.concatenate(target_id,axis=0)
        pixel_mask = np.concatenate(pixel_mask,axis=0)
        mask_pixel = pixel_mask !=0

        diff[mask_pixel] == np.nan
        var[mask_pixel] == np.nan
        spectra[mask_pixel] == np.nan

        return(wavelength,spectra,diff,var,target_id,pixel_mask)
