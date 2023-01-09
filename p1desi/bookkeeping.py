import os, itertools


def return_folder_name(
    region, lines_name, catalog_name, dla_name, bal, noise_estimate, suffix
):
    if (suffix == "None") | (suffix == None):
        suffix = ""
    folder_name = f"{region}_noise{noise_estimate}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix}"
    return folder_name


def return_delta_path(pk_path, folder_name):
    pk = os.path.join(pk_path, f"p1d_{folder_name}", "deltas")
    return pk


def return_pk_path(pk_path, folder_name, snr_cut_path):
    pk = os.path.join(pk_path, f"p1d_{folder_name}", f"pk1d_SNRcut{snr_cut_path}")
    return pk


def return_pk_path_interface(main_config):
    pk_path = main_config.getstr("pk_path")
    use_bookkeeping = main_config.getboolean("use_bookkeeping")
    if use_bookkeeping:
        pk = []
        pk_sb = []
        outname = []
        region = main_config.getliststr("region")
        lines_name = main_config.getliststr("lines_name")
        catalog_name = main_config.getliststr("catalog_name")
        dla_name = main_config.getliststr("dla_name")
        bal = main_config.getlistfloat("bal")
        noise_estimate = main_config.getliststr("noise_estimate")
        suffix = main_config.getliststr("suffix")
        snr_cut_path = main_config.getlistint("snr_cut_path")
        for param in itertools.product(
            region,
            lines_name,
            catalog_name,
            dla_name,
            bal,
            noise_estimate,
            suffix,
            snr_cut_path,
        ):
            folder_name = return_folder_name(*param[:-1])
            outname.append(f"{folder_name}_SNRcut{param[-1]}")
            pk.append(return_pk_path(pk_path, folder_name, param[-1]))

            region_sb = main_config.getstr("region_sb")
            if region_sb is None:
                pk_sb.append(None)
            else:
                folder_name_sb = return_folder_name(region_sb, *param[1:-1])
                pk_sb.append(return_pk_path(pk_path, folder_name_sb, param[-1]))
    else:
        pk = main_config.getliststr("abs_pk_path")
        pk_sb = main_config.getliststr("abs_pk_path_sb")
        outname = main_config.getliststr("outname")
    return (pk, pk_sb, outname)


def return_mean_pk_name(velunits, logsample, snr_cut_mean=None):
    return f'mean_Pk1d{"_log" if logsample else ""}{"_vel" if velunits else ""}{"_snr_cut_mean" if snr_cut_mean is not None else ""}.fits.gz'
