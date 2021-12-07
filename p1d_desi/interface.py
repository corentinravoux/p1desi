import os
import configparser
import ast
import numpy as np
from p1d_desi import p1d_treat,p1d_plot


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

    mean_config = config["compute mean"]
    plot_config = config["plot power"]


    if(main_config.getboolean("compute_mean")):
        compute_mean(pk,mean_config,main_config)

    plot_path = os.path.join(main_path,plot_config.getstr("path_plot"))
    if(main_config.getboolean("compute_mean")):
        plot(pk,plot_path,plot_config,main_config)



def compute_mean(pk,mean_config,main_config):
    print("Treating path: ",pk)
    p1d_treat.compute_Pk_means_parallel(pk,
                                        mean_config.getdict("args_k_array"),
                                        np.array(mean_config.gettuplefloat("zbins")),
                                        searchstr=mean_config.getstr("searchstr"),
                                        ncpu=mean_config.getint("ncpu"),
                                        overwrite=mean_config.getboolean("overwrite"),
                                        velunits=main_config.getboolean("velunits"),
                                        debug=mean_config.getboolean("debug"),
                                        nomedians=mean_config.getboolean("nomedians"),
                                        logsample=mean_config.getboolean("logsample"))


# def plot(pk,plot_path,plot_config,main_config):
#
#     print("Plotting path: ",main_config.getstr("pk_path"))
#     mean_pk = os.path.join(pk,f"mean_Pk1d_par{'_vel' if main_config.getboolean('velunits') else ''}.fits.gz")
#     data = p1d_plot.read_pk_means(mean_pk)
#     p1d_plot.plot_data(data,
#                        zbins,
#                        outname,
#                        plot_P=plot_P,
#                        comparison=comparison,
#                        comparison_model=comparison_model,
#                        comparison_model_file=comparison_model_file,
#                        plot_diff=plot_diff_model,
#                        **kwargs)
#
#     if(substract_sb):
#         pk_means_sb = p1d_plot.read_pk_means(pk_means_name_sb)
#         p1d_plot.plot_data(data,
#                            zbins,
#                            f"{outname}_{region_sb}_substracted",
#                            plot_P=plot_P,
#                            comparison=comparison,
#                            comparison_model=comparison_model,
#                            comparison_model_file=comparison_model_file,
#                            plot_diff=plot_diff_model,
#                            substract_sb=pk_means_sb,
#                            **kwargs)
#
#
#
#     p1d_plot.plot_noise_study(data,
#                               zbins,
#                               outname,
#                               k_units_noise_study,
#                               use_diff_noise,
#                               plot_noise_ratio,
#                               plot_noise_comparison_mean_k,
#                               plot_side_band,
#                               side_band_comp=None,
#                               side_band_legend=["SB1","SB2"],
#                               fit_asymptote_ratio= fit_asymptote_ratio,
#                               **kwargs_noise1)
#
#
#     if(plot_noise_comparison_mean_z):
#         p1d_plot.compute_and_plot_mean_z_noise_power(data,
#                                                      zbins,
#                                                      outname,
#                                                      **kwargs_noise2)
#
#     p1d_treat.compute_Pk_means_parallel(main_config.getstr("pk_path"),
#                                         mean_config.getdict("args_k_array"),
#                                         np.array(mean_config.gettuplefloat("zbins")),
#                                         searchstr=mean_config.getstr("searchstr"),
#                                         ncpu=mean_config.getint("ncpu"),
#                                         overwrite=mean_config.getboolean("overwrite"),
#                                         velunits=mean_config.getboolean("velunits"),
#                                         debug=mean_config.getboolean("debug"),
#                                         nomedians=mean_config.getboolean("nomedians"),
#                                         logsample=mean_config.getboolean("logsample"))
#
#
