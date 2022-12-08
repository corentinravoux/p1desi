import scipy
from p1desi import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def compute_and_plot_mean_z_noise_power(data,
                                        zbins,
                                        outname,
                                        **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)

    dict_noise_diff=compute_mean_z_noise_power(data,zbins,kmin=kmin,kmax=kmax)
    plot_mean_z_noise_power(dict_noise_diff,zbins,outname)



def compute_mean_z_noise_power(data,zbins,kmin=4e-2,kmax=2.5):
    velunits = data.meta["VELUNITS"]

    if velunits and kmax==2:
        kmax=0.035
    if velunits and kmin==4e-2:
        kmin=8e-4

    diff_model = {"Pk_diff" : [] ,"Pk_noise" : []}
    for iz,z in enumerate(zbins):
        dat=data[iz]
        select=dat['N']>0
        k=dat['meank'][select]
        Pk_noise = dat["meanPk_noise"][select][k<kmax]
        Pk_diff = dat["meanPk_diff"][select][k<kmax]
        diff_model["Pk_diff"].append(Pk_diff)
        diff_model["Pk_noise"].append(Pk_noise)

    dict_noise_diff = {"diff":[],"error_diff" : [],"pipeline" : [],"error_pipeline":[],"diff_over_pipeline":[],"error_diff_over_pipeline":[]}
    for i in range(len(diff_model["Pk_noise"])):
        noise_error = scipy.stats.sem(diff_model["Pk_noise"][i],ddof=0)
        diff_error = scipy.stats.sem(diff_model["Pk_diff"][i],ddof=0)
        diff_over_noise_error = (np.mean(diff_model["Pk_diff"][i])/np.mean(diff_model["Pk_noise"][i]))*np.sqrt((diff_error/np.mean(diff_model["Pk_diff"][i]))**2 + (noise_error/np.mean(diff_model["Pk_noise"][i]))**2)
        dict_noise_diff["diff"].append(np.mean(diff_model["Pk_diff"][i]))
        dict_noise_diff["error_diff"].append(diff_error)
        dict_noise_diff["pipeline"].append(np.mean(diff_model["Pk_noise"][i]))
        dict_noise_diff["error_pipeline"].append(noise_error)
        dict_noise_diff["diff_over_pipeline"].append(np.mean((diff_model["Pk_diff"][i] - diff_model["Pk_noise"][i])/diff_model["Pk_noise"][i]))
        dict_noise_diff["error_diff_over_pipeline"].append(diff_over_noise_error)
    dict_noise_diff["zbins"] = zbins

    return(dict_noise_diff)




def plot_mean_z_noise_power(dict_noise_diff,zbins,outname,dreshift = 0.02):
    fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
    ax[1].set_xticks(zbins)
    for i in range(len(dict_noise_diff["pipeline"])):
        scale_fac = 3e5/((1+zbins[i])*1216)
        pipeline = dict_noise_diff["pipeline"][i] * scale_fac
        error_pipeline = dict_noise_diff["error_pipeline"][i] * scale_fac
        diff = dict_noise_diff["diff"][i] * scale_fac
        error_diff = dict_noise_diff["error_diff"][i] * scale_fac
        diff_over_pipeline = ( dict_noise_diff["diff"][i] - dict_noise_diff["pipeline"][i] ) * scale_fac
        error_diff_over_pipeline = diff_over_pipeline * np.sqrt((error_pipeline/pipeline)**2 + (error_diff/diff)**2)
        ax[0].errorbar(zbins[i],pipeline,error_pipeline,marker='*',color =f"C{i}")
        ax[0].errorbar(zbins[i] + dreshift,diff,error_diff,marker='o',color =f"C{i}")
        ax[1].errorbar(zbins[i],diff_over_pipeline,error_diff_over_pipeline,marker='.')
    legend_elements = [Line2D([], [], color='k', marker='*', linestyle='None', label='$P_{pipeline}$'),
                       Line2D([], [], color='k', marker='o', linestyle='None', label='$P_{diff}$')]
    ax[0].legend(handles=legend_elements)
    ax[0].set_ylabel('$<P> [km/s]$')
    ax[1].set_ylabel('$\delta<P> [km/s]$')
    ax[1].set_xlabel('z')
    ax[1].legend(handles = [Line2D([], [], color='k', marker='None', linestyle='None',
                                  label='Average for all redshift = ${}$%'.format(np.round(np.mean(dict_noise_diff["diff_over_pipeline"])*100,2)))],frameon=False)
    plt.tight_layout()
    fig.savefig("{}_mean_ratio_diff_pipeline_power_redshift.pdf".format(outname),format="pdf")



def plot_several_mean_z_noise_power(list_dict,nameout,legend,colors,dreshift = 0.01,marker = [".","^","*","+","x"],obs_wavelength=True):
    marker = marker[0:len(list_dict)]
    fig,ax=plt.subplots(3,1,figsize=(8,8),sharex=True)
    displacement = np.array([(i - len(list_dict)//2 - 0.5*(len(list_dict)%2 -1))*dreshift for i in range(len(list_dict))])
    if(obs_wavelength):
        lya = 1215.673123130217
        zbins = np.round((1+list_dict[0]["zbins"])*lya,0)
        displacement = displacement*lya
        ax[2].set_xlabel('Observed wavelength [$\AA$]')
    else :
        zbins = list_dict[0]["zbins"]
        ax[2].set_xlabel('z')
    ax[2].set_xticks(zbins)
    for i in range(len(list_dict)):
        for j in range(len(list_dict[i]["diff"])):
            ax[0].errorbar(zbins[j] + displacement[i],list_dict[i]["pipeline"][j],list_dict[i]["error_pipeline"][j],marker=marker[i],ecolor=colors[j],color=colors[j])
            ax[1].errorbar(zbins[j] + displacement[i],list_dict[i]["diff"][j],list_dict[i]["error_diff"][j],marker=marker[i],ecolor=colors[j],color=colors[j])
            ax[2].errorbar(zbins[j] + displacement[i],list_dict[i]["diff_over_pipeline"][j],list_dict[i]["error_diff_over_pipeline"][j],marker=marker[i],ecolor=colors[j],color=colors[j])
    legend_elements = [Line2D([0], [0], marker=marker[i], color='k', label=legend[i]) for i in range(len(marker))]
    ax[0].legend(handles=legend_elements)
    ax[0].set_ylabel('$mean_{k}(P_{pipeline}) [\AA]$')
    ax[1].set_ylabel('$mean_{k}(P_{diff}) [\AA]$')
    ax[2].set_ylabel('$mean_{k}((P_{diff}-P_{pipeline})/P_{pipeline})$')
    plt.tight_layout()
    fig.savefig("{}_mean_ratio_diff_pipeline_power_redshift.pdf".format(nameout),format="pdf")



def plot_noise_comparison_function(zbins,
                                   data,
                                   out_name,
                                   mean_dict,
                                   k_units,
                                   **kwargs):

    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)
    fig2,ax2=plt.subplots(4,1,figsize=(8,10),sharex=True)
    for z,d in zip(zbins,data):
        ax2[0].plot(d['meank'],d["meanPk_noise"],label=f'{z:.1f}')
        if(k_units == "A"):
            ax2[0].set_ylabel("$P_{pipeline} [\AA]$")
        elif(k_units == "kms"):
            ax2[0].set_ylabel("$P_{pipeline} [km/s]$")
        ax2[1].plot(d['meank'],d["meanPk_diff"],label=f'{z:.1f}')
        if(k_units == "A"):
            ax2[1].set_ylabel("$P_{diff} [\AA]$")
        elif(k_units == "kms"):
            ax2[1].set_ylabel("$P_{diff} [km/s]$")
        yerr = (d['errorPk_diff'] / d['meanPk_noise']) *  np.sqrt((d['errorPk_noise'] / d['meanPk_noise'])**2 + (d['errorPk_diff'] / d['meanPk_diff'])**2)
        ax2[2].errorbar(d['meank'],(d['meanPk_diff'] -d['meanPk_noise']) / d['meanPk_noise'], yerr = yerr , fmt = 'o')#,marker_size=6)
        ax2[2].set_ylabel('$(P_{diff} - P_{pipeline})/P_{pipeline}$')
        ax2[0].legend()
    ax2[3].errorbar(mean_dict["k_array"],(mean_dict["meanPk_diff"] - mean_dict["meanPk_noise"]) / mean_dict["meanPk_noise"], yerr =mean_dict["error_diffovernoise"], fmt = 'o')#,marker_size=6)
    ax2[3].set_ylabel('$mean_{z}((P_{diff} - P_{pipeline})/P_{pipeline})$')
    if(k_units == "A"):
        ax2[3].set_xlabel('k[1/$\AA$]')
        utils.place_k_speed_unit_axis(fig2,ax2[0])
    elif(k_units == "kms"):
        ax2[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax2[0].set_xlim(kmin,kmax)
    fig2.tight_layout()
    fig2.savefig(f"{out_name}_ratio_diff_pipeline_power_wavevector_unit{k_units}.pdf",format="pdf")


def plot_noise_power_ratio(data,
                           zbins,
                           out_name,
                           mean_dict,
                           noise_to_plot,
                           labelnoise,
                           k_units,
                           fit_asymptote= False,
                           plot_difference = False,
                           **kwargs):



    save_txt = utils.return_key(kwargs,"save_txt",True)
    kmin = utils.return_key(kwargs,"kmin",None)
    kmax = utils.return_key(kwargs,"kmax",None)
    ncol_legend = utils.return_key(kwargs,"ncol_legend",2)
    ratio_y_min = utils.return_key(kwargs,"ratio_y_min",0.5)
    ratio_y_max = utils.return_key(kwargs,"ratio_y_max",1.1)
    diff_y_min = utils.return_key(kwargs,"diff_y_min",-0.01)
    diff_y_max = utils.return_key(kwargs,"diff_y_max",0.01)
    fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)
    txt_to_save = []
    for z,d in zip(zbins,data):
        ax[0].plot(d['meank'],d['meanPk_raw'],label=f'{z:.1f} ({d["N_chunks"]})')
        if(k_units == "A"):
            ax[0].set_ylabel('$P_{raw} [\AA]$')
        elif(k_units == "kms"):
            ax[0].set_ylabel('$P_{raw} [km/s]$')
        ax[0].legend(ncol=ncol_legend)
        ax[1].plot(d['meank'],d[noise_to_plot],label=f'{z:.1f}')
        if(k_units == "A"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [\AA]$')
        elif(k_units == "kms"):
            ax[1].set_ylabel('$P_{' + labelnoise +'} [km/s]$')
        if(plot_difference):
            ax[2].plot(d['meank'], d['meanPk_raw'] - d[noise_to_plot],label=f'{z:.1f}')
            ax[2].set_ylabel('$P_{raw} - P_{' + labelnoise +'}$')
        else:
            ax[2].plot(d['meank'],d[noise_to_plot]/d['meanPk_raw'],label=f'{z:.1f}')
            ax[2].set_ylabel('$P_{' + labelnoise +'}/P_{raw}$')
        if(save_txt):
            z_array = np.full(d['meank'].shape,z)
            txt_to_save.append(np.stack([z_array,d['meank'],d['meanPk_raw'],d[noise_to_plot]],axis=1))

    txt_to_save = np.concatenate(txt_to_save,axis=0)
    np.savetxt(f"{out_name}_ratio_{labelnoise}_raw_power_unit{k_units}.txt",txt_to_save,header = f"z k Pk_raw {labelnoise}")

    if(plot_difference):
        ax[3].plot(mean_dict["k_array"],mean_dict["meanPk_raw"] - mean_dict[noise_to_plot], marker = 'o')
        ax[3].set_ylabel('$mean_{z}(P_{raw} - P_{' + labelnoise +'})$')
    else:
        ax[3].errorbar(mean_dict["k_array"],mean_dict[noise_to_plot]/mean_dict["meanPk_raw"], yerr =mean_dict["error_{}overraw".format(noise_to_plot)], fmt = 'o')#,marker_size=6)
        ax[3].set_ylabel('$mean_{z}(P_{' + labelnoise +'}/P_{raw})$')
    legend = []
    if(fit_asymptote):
        try :
            f_const = lambda x,a : np.array([a for i in range(len(x))])
            if(np.min(mean_dict["k_array"])> 3.0) : kmin_fit = np.min(mean_dict["k_array"])
            else : kmin_fit = 3.0
            mask = mean_dict["k_array"] > kmin_fit
            if plot_difference :
                init_value = 0
                fit = (mean_dict["meanPk_raw"] - mean_dict[noise_to_plot])[mask]
            else :
                init_value = 1
                fit = (mean_dict[noise_to_plot]/mean_dict["meanPk_raw"])[mask]
            arg_func = scipy.optimize.curve_fit(f_const,
                                                mean_dict["k_array"][mask],
                                                fit,
                                                p0=[init_value])
            cont_k_array = np.linspace(kmin_fit,np.max(mean_dict["k_array"]),500)
            fit_exp = f_const(cont_k_array,*arg_func[0])
            fit_exp_min = f_const(cont_k_array,*(arg_func[0]-np.diag(arg_func[1])))
            fit_exp_max = f_const(cont_k_array,*(arg_func[0]+np.diag(arg_func[1])))
            ax[3].fill_between(cont_k_array, fit_exp_min, fit_exp_max,facecolor='grey', interpolate=True,alpha=0.5)
            ax[3].plot(cont_k_array,fit_exp)
            legend.append("asymptote = {}".format(np.round(arg_func[0][0],5)))
        except:
            print("Pdiff over Praw fit did not converge")
    mask_k = mean_dict["k_array"] > 3.10

    legend = legend + [f'alpha = {np.mean((mean_dict["meanPk_raw"] - mean_dict[noise_to_plot])[mask_k])}',
                        f'beta = {np.mean((mean_dict[noise_to_plot]/mean_dict["meanPk_raw"])[mask_k])}']
    ax[3].legend(legend)
    if(k_units == "A"):
        ax[3].set_xlabel('k[1/$\AA$]')
        utils.place_k_speed_unit_axis(fig,ax[0])
    elif(k_units == "kms"):
        ax[3].set_xlabel('k[$s/km$]')
    if(kmin is not None): ax[0].set_xlim(kmin,kmax)
    for i in [0,1,2,3]:
        ax[i].grid()
    if(plot_difference):
        ax[2].set_ylim(diff_y_min,diff_y_max)
        ax[3].set_ylim(diff_y_min,diff_y_max)
    else:
        ax[2].set_ylim(ratio_y_min,ratio_y_max)
        ax[3].set_ylim(ratio_y_min,ratio_y_max)
    fig.tight_layout()
    if(plot_difference):
        out_name = f"{out_name}_power_difference"
    fig.savefig(f"{out_name}_ratio_{labelnoise}_raw_power_unit{k_units}.pdf",format="pdf")



def plot_noise_study(data,
                     zbins,
                     out_name,
                     k_units,
                     use_diff_noise,
                     plot_noise_ratio,
                     plot_noise_comparison,
                     plot_side_band,
                     side_band_comp=None,
                     side_band_legend=["SB1","SB2"],
                     fit_asymptote_ratio= False,
                     plot_difference = False,
                     **kwargs):


    mean_dict = return_mean_z_dict(zbins,data)

    if(use_diff_noise):
        noise_to_plot,labelnoise = 'meanPk_diff','diff'
    else:
        noise_to_plot,labelnoise = 'meanPk_noise','pipeline'
    if(plot_noise_ratio):
        plot_noise_power_ratio(data,
                               zbins,
                               out_name,
                               mean_dict,
                               noise_to_plot,
                               labelnoise,
                               k_units,
                               fit_asymptote= fit_asymptote_ratio,
                               plot_difference = plot_difference,
                               **kwargs)
    if(plot_noise_comparison):
        plot_noise_comparison_function(zbins,
                                       data,
                                       out_name,
                                       mean_dict,
                                       k_units,
                                       **kwargs)

