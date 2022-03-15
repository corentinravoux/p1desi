
import logging



dict_picca_delta = {"mode" : "desi_healpix",
                    "region": "lya",
                    }



def run_picca_delta(dict_picca):
    cmd = " picca_deltas.py"
    cmd += f" --in-dir {path_in}"
    cmd += f" --drq {cat}"
    cmd += f" --out-dir {path_out_deltas}"
    cmd += f" --nproc ${nb_proc}"
    for key,value in dict_picca.items():
        cmd += f" --{key} {value}"


    --lambda-rest-min ${lrestmin} \
    --lambda-rest-max ${lrestmax} \
    ${use_constant_weight_flag} \
    ${coadd_picca_flag} \
    ${compute_diff_flux_flag} \
    ${dla_flag} \
    ${bal_flag} \
    --min-SNR ${snr_cut} \
    --mask-file ${lines} \
    --log ${path_log}/log_deltas_${folder_name} \
    --iter ${path_log}/iter_deltas_${folder_name} > ${path_log}/log_deltas_out_${folder_name} 2> ${path_log}/log_deltas_err_${folder_name}

    default_picca_delta(cmd)

    subprocess.call(cmd, shell=True)




def return_region(region):
    if(region == "lya"):
        lrestmin=1040.0
        lrestmax=1200.0
    elif(region == "SB1"):
        lrestmin=1270.0
        lrestmax=1380.0
    elif(region == "SB2"):
        lrestmin=1410.0
        lrestmax=1520.0
    else:
        raise ValueError(f"No region corresponds to {region}")
    return(lrestmin,lrestmax)


def default_picca_delta(cmd):
    cmd += " --delta-format Pk1D"
    cmd += " --use-desi-P1d-changes"
    cmd += " --order 0"
    cmd += " --mc-rebin-fac 4"
    cmd += " --rebin 1"
    cmd += " --lambda-min 3500."
    cmd += " --lambda-max 7500.0"
    return(cmd)




def run_picca_delta_extraction(dict_picca):
    cmd = " picca_delta_extraction.py"
    subprocess.call(cmd, shell=True)



def run_export_pk1d(input,output):
    cmd = " picca_export.py"
    cmd += f" --data {input}"
    cmd += f" --out {output}"
    subprocess.call(cmd, shell=True)




picca_deltas.py \

mkdir -p ${path_out}/pk1d_${noise_estimate}_noise_estimate

picca_Pk1D.py \
--in-dir ${path_out}/deltas \
--in-format fits \
--out-dir ${path_out}/pk1d_${noise_estimate}_noise_estimate \
--lambda-obs-min 3750. \
--noise-estimate ${noise_estimate} \
--pixel-correction inverse \
--nproc ${nb_proc} \
--use-desi-new-defaults \
--nb-pixel-masked-max 120 \
--nb-part 3 \
--SNR-min ${snr_cut} \
--nb-noise-exp 2500 > ${path_log}/log_p1d_out_${folder_name} 2> ${path_log}/log_p1d_err_${folder_name}
