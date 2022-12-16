import pickle, os, fitsio, glob
import numpy as np
from functools import partial
from p1desi import metals, utils


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


def apply_correction(pk,file_correction,type_correction):
    A_corr = eval(f"prepare_{type_correction}_correction")(pk.zbin,file_correction)
    for z in pk.zbin:
        pk.p[z] = pk.p[z] * A_corr[z](pk.k[z])
        pk.norm_p[z] = pk.norm_p[z] * A_corr[z](pk.k[z])
        pk.err[z] = pk.err[z] * A_corr[z](pk.k[z])
        pk.norm_err[z] = pk.norm_err[z] * A_corr[z](pk.k[z])



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

def apply_correction_eboss(pk,file_correction,type_correction):
    A_corr = eval(f"prepare_{type_correction}_correction_eboss")(pk.zbin,file_correction)
    for z in pk.zbin:
        pk.p[z] = pk.p[z] * A_corr[z](pk.k[z])
        pk.norm_p[z] = pk.norm_p[z] * A_corr[z](pk.k[z])
        pk.err[z] = pk.err[z] * A_corr[z](pk.k[z])
        pk.norm_err[z] = pk.norm_err[z] * A_corr[z](pk.k[z])



###################################################
############## Side band corrections ##############
###################################################


def prepare_metal_subtraction(zbins,file_metal,velunits=False):
    (param_SB1_mean,
     param_SB1_indiv)= pickle.load(open(file_metal,"rb"))
    P_metal_m = {}
    for iz,z in enumerate(zbins):
        if velunits:
            P_metal_m[z] = partial(metals.model_SB1_indiv_kms,param_SB1_mean,param_SB1_indiv[iz][0],param_SB1_indiv[iz][1])
        else:
            P_metal_m[z] = partial(metals.model_SB1_indiv,param_SB1_mean,z,param_SB1_indiv[iz][0],param_SB1_indiv[iz][1])
    return P_metal_m



def subtract_metal(pk,file_metal):
    P_metal_m = prepare_metal_subtraction(pk.zbin,file_metal,velunits=pk.velunits)
    for z in pk.zbin:
        pk.p[z] = pk.p[z] - P_metal_m[z](pk.k[z])
        pk.norm_p[z] = pk.norm_p[z]  - (pk.k[z] * P_metal_m[z](pk.k[z]) / np.pi)


def prepare_metal_correction_eboss(file_metal_eboss):
    param_sb_eboss = np.loadtxt(file_metal_eboss)
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



def apply_p1d_corections(pk,
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


    if apply_DESI_maskcont_corr & apply_eBOSS_maskcont_corr:
        return KeyError("Same type of correction is applied two times")

    if apply_DESI_sb_corr & apply_eBOSS_sb_corr:
        return KeyError("Same type of correction is applied two times")


    if apply_DESI_maskcont_corr:
        for type_correction in correction_to_apply:
            apply_correction(pk,eval(f"file_correction_{type_correction}"),type_correction)

    if apply_eBOSS_maskcont_corr:
        for type_correction in correction_to_apply:
            apply_correction_eboss(pk,eval(f"file_correction_{type_correction}_eboss"),type_correction)

    if apply_DESI_sb_corr:
        subtract_metal(pk,file_metal)

    if apply_eBOSS_sb_corr:
        subtract_metal_eboss(pk,file_metal_eboss)

    return pk





###################################################
################ Noise corrections ################
###################################################



def model_noise_correction(SNR,A,g,B):
    power_law = A * SNR**(-g) + B
    return(power_law)



def correct_individual_pk_noise(pk_in,
                                pk_out,
                                qsocat,
                                correction):
    os.makedirs(pk_out,exist_ok=True)
    pk_files = glob.glob(os.path.join(pk_in,"Pk1D-*"))

    qso_file = fitsio.FITS(qsocat)["QSO_CAT"]
    targetid_qso = qso_file["TARGETID"][:]
    survey_qso = qso_file["SURVEY"][:]
    for i in range(len(pk_files)):
        pk_out_name = pk_files[i].split("/")[-1]
        f_out = fitsio.FITS(os.path.join(pk_out,pk_out_name),'rw',clobber=True)
        f = fitsio.FITS(pk_files[i])
        for j in range(1,len(f)):
            header = dict(f[j].read_header())
            snr = header["MEANSNR"]
            id = header["LOS_ID"]
            survey_id = survey_qso[np.argwhere(targetid_qso == id)[0]][0]
            p_noise_miss = model_noise_correction(snr,*correction[survey_id])
            line = f[j].read()
            new_line = np.zeros(line.size, dtype=[(line.dtype.names[i],line.dtype[i]) for i in range(len(line.dtype))] + [('PK_NOISE_MISS','>f8')])
            for name in line.dtype.names :
                new_line[name] = line[name]
            new_line['PK_NOISE'] = new_line['PK_NOISE'] + p_noise_miss
            new_line['PK_NOISE_MISS'] = np.full(line.size,p_noise_miss)
            f_out.write(new_line,header=header,extname=str(id))
