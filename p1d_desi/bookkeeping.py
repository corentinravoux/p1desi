import os


def desi_data_keeping(main_config):
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
        region_sb = main_config.getstr("region_sb")

        folder_name=f"{region}_SNRcut{snr_cut}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix_flag}"
        folder_name_sb=f"{region_sb}_SNRcut{snr_cut}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix_flag}"
    print(folder_name)

    path_abs = main_config.getstr("path_abs")

    # folder_name=${region}_SNRcut${snr_cut}_lines${lines_name}_cat${catalog_name}_dla${dla_name}_bal${bal}${suffix_flag}
    # path_to_pk=${path_in}/p1d_${folder_name}/pk1d_${noise_estimate}_noise_estimate
    #
    # folder_name_sb=${region_sb}_SNRcut${snr_cut}_lines${lines_name}_cat${catalog_name}_dla${dla_name}_bal${bal}${suffix_flag}
    # path_to_pk_sb=${path_in}/p1d_${folder_name_sb}/pk1d_${noise_estimate}_noise_estimate
    #
    # path_plot=${path_plot}/${region}/p1d_${folder_name}_${noise_estimate}_noise_estimate
    #
    #
