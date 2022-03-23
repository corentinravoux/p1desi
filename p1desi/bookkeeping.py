import os

def return_folder_name(region,lines_name,catalog_name,dla_name,bal,noise_estimate,suffix):
    if(suffix == "None")|(suffix == None):
        suffix = ""
    folder_name=f"{region}_noise{noise_estimate}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix}"
    return(folder_name)

def return_delta_path(pk_path,folder_name):
    pk = os.path.join(pk_path,f"p1d_{folder_name}","deltas")
    return(pk)

def return_pk_path(pk_path,folder_name,snr_cut):
    pk = os.path.join(pk_path,f"p1d_{folder_name}",f"pk1d_SNRcut{snr_cut}")
    return(pk)

def return_pk_path_interface(main_config):
    pk_path = main_config.getstr("pk_path")
    use_bookkeeping = main_config.getboolean("use_bookkeeping")
    if(use_bookkeeping):
        region = main_config.getstr("region")
        lines_name = main_config.getstr("lines_name")
        catalog_name = main_config.getstr("catalog_name")
        dla_name = main_config.getstr("dla_name")
        bal = main_config.getstr("bal")
        noise_estimate = main_config.getstr("noise_estimate")
        suffix = main_config.getstr("suffix")

        folder_name=return_folder_name(region,lines_name,catalog_name,dla_name,bal,noise_estimate,suffix)

        snr_cut = main_config.getint("snr_cut")
        outname = f"{folder_name}_SNRcut{snr_cut}"
        pk = return_pk_path(pk_path,folder_name,snr_cut)

        region_sb = main_config.getstr("region_sb")
        if(region_sb is None):
            pk_sb = None
        else:
            folder_name_sb=return_folder_name(region_sb,snr_cut,lines_name,catalog_name,dla_name,bal,noise_estimate,suffix)
            pk_sb = return_pk_path(pk_path,folder_name_sb,snr_cut)
    else:
        pk = main_config.getstr("abs_pk_path")
        pk_sb = main_config.getstr("abs_pk_path_sb")
        outname = main_config.getstr("outname")

    return(pk,pk_sb,outname)


def return_mean_pk_name(velunits,logsample):
    return(f'mean_Pk1d{"_log" if logsample else ""}{"_vel" if velunits else ""}.fits.gz')
