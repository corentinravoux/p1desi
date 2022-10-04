import pickle,os,fitsio
import numpy as np
from functools import partial
from p1desi import plotpk, metals


###################################################
######### Corrections masking + continuum #########
###################################################


################ DESI corrections #################

def prepare_hcd_correction(zbins,file_correction_hcd):
    param_hcd = pickle.load(open(file_correction_hcd,"rb"))
    A_hcd = {}
    for iz,z in enumerate(zbins):
        A_hcd[z] = np.poly1d(param_hcd[iz])
    return A_hcd

def prepare_lines_correction(zbins,file_correction_lines):
    param_lines = pickle.load(open(file_correction_lines,"rb"))
    A_lines = {}
    for iz,z in enumerate(zbins):
        A_lines[z] = np.poly1d(param_lines[iz])
    return(A_lines)

def model_cont_correction(a0,a1,a2,k):
    return a0*np.exp(-k/a1) + a2 + 1

def prepare_cont_correction(zbins,file_correction_cont):
    param_cont = pickle.load(open(file_correction_cont,'rb'))
    A_cont = {}
    for iz,z in enumerate(zbins):
        A_cont[z] = partial(model_cont_correction,*param_cont[iz])
    return(A_cont)


def apply_correction(dict_plot,zbins,file_correction,type_correction):
    A_corr = eval(f"prepare_{type_correction}_correction")(zbins,file_correction)
    for iz,z in enumerate(zbins):
        dict_plot[z]["p_to_plot"] =  dict_plot[z]["p_to_plot"] * A_corr[z](dict_plot[z]["k_to_plot"])



################ eBOSS corrections ################



def prepare_hcd_correction_eboss(zbins,hcd_eboss_name):
    param_hcd_eboss = np.loadtxt(hcd_eboss_name)
    A_hcd = {}
    for iz,z in enumerate(zbins):
        A_hcd[z] = np.poly1d(param_hcd_eboss[iz])
    return A_hcd


def prepare_lines_correction_eboss(zbins,lines_eboss_name):
    param_lines_eboss = np.loadtxt(lines_eboss_name)
    A_lines = {}
    for iz,z in enumerate(zbins):
        A_lines[z] = np.poly1d(param_lines_eboss[iz])
    return A_lines



def model_cont_correction_eboss(a0,a1,k):
    return (a0/k)+a1


def prepare_cont_correction_eboss(zbins,cont_eboss_name):
    param_cont_eboss = np.loadtxt(cont_eboss_name)
    A_cont = {}
    for iz,z in enumerate(zbins):
        if iz < 6:
            A_cont[z] = partial(model_cont_correction_eboss,param_cont_eboss[iz][0],param_cont_eboss[iz][1])
        else:
            A_cont[z] = np.poly1d(param_cont_eboss[iz])
    return A_cont

def apply_correction_eboss(dict_plot,zbins,file_correction,type_correction):
    A_corr = eval(f"prepare_{type_correction}_correction_eboss")(zbins,file_correction)
    for iz,z in enumerate(zbins):
        k_speed = dict_plot[z]["k_to_plot"]/(utils.speed_light/(1+z)/utils.lambdaLy)
        dict_plot[z]["p_to_plot"] =  dict_plot[z]["p_to_plot"] * A_corr[z](k_speed)




###################################################
############## Side band corrections ##############
###################################################


def prepare_metal_subtraction(zbins,file_metal):
    (param_SB1_mean,
     param_SB1_indiv)= pickle.load(open(file_metal,"rb"))
    P_metal_m = {}
    for iz,z in enumerate(zbins):
        P_metal_m[z] = partial(metals.model_SB1_indiv,param_SB1_mean,z,param_SB1_indiv[iz][0],param_SB1_indiv[iz][1])
    return P_metal_m



def subtract_metal(dict_plot,zbins,file_metal,plot_P):
    P_metal_m = prepare_metal_subtraction(zbins,file_metal)
    for iz,z in enumerate(zbins):
        if(plot_P):
            dict_plot[z]["p_to_plot"] = dict_plot[z]["p_to_plot"] - P_metal_m[z](dict_plot[z]["k_to_plot"])
        else:
            dict_plot[z]["p_to_plot"] = dict_plot[z]["p_to_plot"] - (dict_plot[z]["k_to_plot"] * P_metal_m[z](dict_plot[z]["k_to_plot"]) / np.pi)




def prepare_metal_correction_eboss(file_metal_eboss):
    sb_eboss ='/global/cfs/cdirs/desi/users/ravouxco/pk1d/eboss_data/LyaSDSS/Data/Correction_SB_Pk_1270_1380_no_BAL.txt'
    param_sb_eboss = np.loadtxt(sb_eboss)
    return param_sb_eboss




def subtract_metal_eboss(dict_plot,zbins,file_metal_eboss,plot_P):
    param_sb_eboss = prepare_metal_correction_eboss(file_metal_eboss)
    nbin_eboss=35
    klimInf_eboss=0.000813
    klimSup_eboss=klimInf_eboss + nbin_eboss*0.000542

    for iz,z in enumerate(zbins):
        mask_sb = param_sb_eboss[:,0] == z
        k_sb_eboss = param_sb_eboss[:,1][mask_sb]
        sb_eboss = param_sb_eboss[:,2][mask_sb]

        k_to_plot = dict_plot[z]["k_to_plot"]

        deltak_eboss = (klimSup_eboss-klimInf_eboss)/nbin_eboss
        ib_eboss =((k_to_plot-klimInf_eboss)/deltak_eboss).astype(int)
        cor_sb = np.zeros(k_to_plot.shape)
        slope_eboss = np.zeros(k_to_plot.shape)

        mask_k_sb_1 = k_to_plot < klimInf_eboss
        slope_eboss[mask_k_sb_1] = (sb_eboss[1]-sb_eboss[0])/deltak_eboss
        cor_sb[mask_k_sb_1] = sb_eboss[0] + slope_eboss[mask_k_sb_1] * (k_to_plot[mask_k_sb_1] - deltak_eboss/2 - klimInf_eboss)

        mask_k_sb_2 = (ib_eboss < nbin_eboss -1)&(~mask_k_sb_1)
        slope_eboss[mask_k_sb_2] = (sb_eboss[ib_eboss[mask_k_sb_2]+1] - sb_eboss[ib_eboss[mask_k_sb_2]])/deltak_eboss
        cor_sb[mask_k_sb_2] = sb_eboss[ib_eboss[mask_k_sb_2]] + slope_eboss[mask_k_sb_2] * (k_to_plot[mask_k_sb_2]- deltak_eboss/2 -(klimInf_eboss + ib_eboss[mask_k_sb_2]*deltak_eboss))

        mask_k_sb_3 = (~mask_k_sb_2)&(~mask_k_sb_1)
        slope_eboss[mask_k_sb_3] = (sb_eboss[-1]-sb_eboss[-2])/deltak_eboss
        cor_sb[mask_k_sb_3] = sb_eboss[-1] + slope_eboss[mask_k_sb_3] * (k_to_plot[mask_k_sb_3]-deltak_eboss/2-(klimSup_eboss-deltak_eboss))

        if(plot_P):
            dict_plot[z]["p_to_plot"] = dict_plot[z]["p_to_plot"] - cor_sb
        else:
            dict_plot[z]["p_to_plot"] = dict_plot[z]["p_to_plot"] - dict_plot[z]["k_to_plot"] * cor_sb/ np.pi



def apply_p1d_corections(mean_pk,
                         zbins,
                         plot_P,
                         z_binsize,
                         velunits,
                         apply_DESI_maskcont_corr,
                         apply_eBOSS_maskcont_corr,
                         apply_DESI_sb_corr,
                         apply_eBOSS_sb_corr,
                         correction_to_apply,
                         file_correction_hcd = None,
                         file_correction_hcd_eboss = None,
                         file_correction_lines = None,
                         file_correction_lines_eboss = None,
                         file_correction_cont = None,
                         file_correction_cont_eboss = None,
                         file_metal = None,
                         file_metal_eboss = None):



    dict_plot = plotpk.prepare_plot_values(mean_pk,
                                           zbins,
                                           plot_P=plot_P,
                                           z_binsize=z_binsize,
                                           velunits=velunits)


    if apply_DESI_maskcont_corr & apply_eBOSS_maskcont_corr:
        return KeyError("Same type of correction is applied two times")

    if apply_DESI_sb_corr & apply_eBOSS_sb_corr:
        return KeyError("Same type of correction is applied two times")


    if apply_DESI_maskcont_corr:
        for type_correction in correction_to_apply:
            apply_correction(dict_plot,zbins,eval(f"file_correction_{type_correction}"),type_correction)

    if apply_eBOSS_maskcont_corr:
        for type_correction in correction_to_apply:
            apply_correction_eboss(dict_plot,zbins,eval(f"file_correction_{type_correction}_eboss"),type_correction)

    if apply_DESI_sb_corr:
        subtract_metal(dict_plot,zbins,file_metal,plot_P)

    if apply_eBOSS_sb_corr:
        subtract_metal_eboss(dict_plot,zbins,file_metal_eboss,plot_P)

    return dict_plot
