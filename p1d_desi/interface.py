
import os
import configparser
import ast
import numpy as np
from p1d_desi import treatpk,plotpk


def parse_int_tuple(input):
    if(input == "None"):
        return(None)
    else:
        return tuple(int(k.strip()) for k in input.strip().split(','))

def parse_float_tuple(input):
    if(input == "None"):
        return(None)
    else:
        return tuple(float(k.strip()) for k in input.strip().split(','))

def parse_str_tuple(input):
    if(input == "None"):
        return(None)
    else:
        return tuple(str(k.strip()) for k in input.strip().split(','))

def parse_dict(input):
    if(input == "None"):
        return(None)
    else:
        acceptable_string = input.replace("'", "\"")
        return(ast.literal_eval(acceptable_string))

def parse_float(input):
    if(input == "None"):
        return(None)
    else:
        return(float(input))

def parse_int(input):
    if(input == "None"):
        return(None)
    else:
        return(int(input))

def parse_string(input):
    if(input == "None"):
        return(None)
    else:
        return(str(input))




def main(input_file):
    config = configparser.ConfigParser(allow_no_value=True,
                                       converters={"str": parse_string,
                                                   "int": parse_int,
                                                   "float": parse_float,
                                                   "tupleint": parse_int_tuple,
                                                   "tuplefloat": parse_float_tuple,
                                                   "tuplestr": parse_str_tuple,
                                                   "dict":parse_dict})
    config.optionxform = lambda option: option
    config.read(input_file)

    main_config = config["main"]
    main_path = os.path.abspath(main_config["path"])
    os.makedirs(main_path,exist_ok=True)
    pk_path = main_config.getstr("pk_path")

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

    pk = os.path.join(pk_path,f"p1d_{folder_name}",f"pk1d_{noise_estimate}_noise_estimate")
    pk_sb = os.path.join(pk_path,f"p1d_{folder_name_sb}",f"pk1d_{noise_estimate}_noise_estimate")

    mean_config = config["compute mean"]
    plot_config = config["plot power"]
    plot_noise_config = config["plot noise"]



    if(main_config.getboolean("compute_mean")):
        compute_mean(pk,mean_config,main_config)

    path_plot = os.path.join(main_path,plot_config.getstr("path_plot"))
    os.makedirs(path_plot,exist_ok=True)
    if(main_config.getboolean("plot_power")):
        plot(pk,pk_sb,path_plot,plot_config,main_config)

    path_plot_noise = os.path.join(main_path,plot_noise_config.getstr("path_plot_noise"))
    os.makedirs(path_plot_noise,exist_ok=True)
    if(main_config.getboolean("plot_noise")):
        plot_noise(pk,path_plot_noise,plot_noise_config,main_config)

def compute_mean(pk,mean_config,main_config):
    print("Treating path: ",pk)
    treatpk.compute_Pk_means_parallel(pk,
                                      mean_config.getdict("args_k_array"),
                                      np.array(mean_config.gettuplefloat("zbins")),
                                      searchstr=mean_config.getstr("searchstr"),
                                      ncpu=mean_config.getint("ncpu"),
                                      overwrite=mean_config.getboolean("overwrite"),
                                      velunits=main_config.getboolean("velunits"),
                                      debug=mean_config.getboolean("debug"),
                                      nomedians=mean_config.getboolean("nomedians"),
                                      logsample=mean_config.getboolean("logsample"))


def plot(pk,pk_sb,path_plot,plot_config,main_config):
    print("Plotting path: ",pk)
    mean_pk = os.path.join(pk,f"mean_Pk1d_par{'_vel' if main_config.getboolean('velunits') else ''}.fits.gz")
    mean_pk_sb = os.path.join(pk_sb,f"mean_Pk1d_par{'_vel' if main_config.getboolean('velunits') else ''}.fits.gz")
    velunits = main_config.getboolean("velunits")
    comparison_str = plot_config.getstr("comparison_str")
    substract_sb = plot_config.getboolean("substract_sb")
    region_sb = plot_config.getstr("region_sb")
    plot_args = plot_config.getdict("plot_args")

    outname = os.path.join(path_plot,f"p1d_model{comparison_str}_unit{'kms' if velunits else 'A'}")

    if(plot_config.getboolean("plot_pk")):
        data = plotpk.read_pk_means(mean_pk)
        if(substract_sb):
            substract_data = mean_pk_sb
            outname =  f"{outname}_{region_sb}_substracted"
        else:
            substract_data = None
        plotpk.plot_data(data,
                         np.array(plot_config.gettuplefloat("zbins_plot")),
                         outname,
                         plot_P=plot_config.getboolean("plot_P"),
                         comparison=plot_config.getstr("pk_comparison"),
                         comparison_model=plot_config.getstr("model_comparison"),
                         comparison_model_file=list(plot_config.gettuplestr("comparison_model_file")),
                         plot_diff=plot_config.getboolean("plot_diff_model"),
                         substract_sb=substract_data,
                         **plot_args)


def plot_noise(pk,path_plot_noise,plot_noise_config,main_config):
    # CR - move side_band in a metals study plotting
    print("Plotting path: ",pk)
    mean_pk = os.path.join(pk,f"mean_Pk1d_par{'_vel' if main_config.getboolean('velunits') else ''}.fits.gz")
    velunits = main_config.getboolean("velunits")
    outname = os.path.join(path_plot_noise,f"p1d_unit{'kms' if velunits else 'A'}")
    zbins_plot = np.array(plot_noise_config.gettuplefloat("zbins_plot"))

    plot_args_noise = plot_noise_config.getdict("plot_args_noise")


    data = plotpk.read_pk_means(mean_pk)

    if(plot_noise_config.getboolean("plot_noise_study")):
        plotpk.plot_noise_study(data,
                                zbins_plot,
                                outname,
                                plot_noise_config.getstr("k_units_noise_study"),
                                plot_noise_config.getboolean("plot_noise_ratio"),
                                plot_noise_config.getboolean("use_diff_noise"),
                                plot_noise_config.getboolean("plot_noise_comparison_mean_k"),
                                plot_noise_config.getboolean("plot_side_band"),
                                side_band_comp=plot_noise_config.getstr("side_band_comp"),
                                side_band_legend=plot_noise_config.gettuplestr("side_band_legend"),
                                fit_asymptote_ratio= plot_noise_config.getboolean("fit_asymptote_ratio"),
                                **plot_args_noise)


    if(plot_noise_config.getboolean("plot_noise_comparison_mean_z")):
        plotpk.compute_and_plot_mean_z_noise_power(data,
                                                   zbins_plot,
                                                   outname,
                                                   **plot_args_noise)
