import os




def path_pk(region,snr_cut,lines_name,catalog_name,dla_name,bal,suffix_flag):
    folder_name=f"{region}_SNRcut{snr_cut}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix_flag}"
    return(folder_name)


def desi_data_keeping(main_config):
    pk_path = main_config.getstr("pk_path")
    use_bookkeeping = main_config.getboolean("use_bookkeeping")
    if(use_bookkeeping):
        region = main_config.getstr("region")
        snr_cut = main_config.getint("snr_cut")
        lines_name = main_config.getstr("lines_name")
        catalog_name = main_config.getstr("catalog_name")
        dla_name = main_config.getstr("dla_name")
        bal = main_config.getstr("bal")
        suffix_flag = main_config.getstr("suffix_flag")
        noise_estimate = main_config.getstr("noise_estimate")


        folder_name=f"{region}_SNRcut{snr_cut}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix_flag}"
        pk = os.path.join(pk_path,f"p1d_{folder_name}",f"pk1d_{noise_estimate}_noise_estimate")


        region_sb = main_config.getstr("region_sb")

        if(region_sb is None):
            pk_sb = None
        else:
            folder_name_sb=f"{region_sb}_SNRcut{snr_cut}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix_flag}"
            pk_sb = os.path.join(pk_path,f"p1d_{folder_name_sb}",f"pk1d_{noise_estimate}_noise_estimate")

    else:
        pk = main_config.getstr("abs_pk_path")
        pk_sb = main_config.getstr("abs_pk_path_sb")

    return(pk,pk_sb)
