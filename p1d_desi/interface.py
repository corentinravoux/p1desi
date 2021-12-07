import os
import configparser
import ast
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

    xcorr_config = config["xcorr"]
    xcorr_plot_config = config["xcorr plot"]



    xcorr_path = os.path.join(main_path,xcorr_config.getstr("xcorr_path"))
    if(main_config.getboolean("compute_mean")):
        compute_mean(xcorr_path,xcorr_config,main_config)

    plot_xcorr_path = os.path.join(main_path,xcorr_plot_config.getstr("plot_xcorr_path"))
    if(main_config.getboolean("compute_mean")):
        plot_xcorr(plot_xcorr_path,xcorr_path,xcorr_plot_config,xcorr_config,main_config)





def compute_mean(xcorr_path,xcorr_config,main_config):
    os.makedirs(xcorr_path,exist_ok=True)


    data=p1d_treat.compute_Pk_means_parallel(path_to_pk,
                                             args,
                                             zbins,
                                             searchstr=searchstr,
                                             ncpu=ncpu,
                                             overwrite=overwrite,
                                             velunits=velunits,
                                             debug=debug,
                                             nomedians=nomedians,
                                             logsample=logsample)


    xcorr_name = os.path.join(xcorr_path,main_config["name"])
    dict_picca_rmu = xcorr_config.getdict("dict_picca_rmu")
    dict_picca_rprt = xcorr_config.getdict("dict_picca_rprt")

    xcorr_dict = {}
    xcorr_dict["drq"] = main_config.getstr("void_catalog")
    xcorr_dict["in-dir"] = main_config.getstr("delta_path")
    xcorr_dict["z-cut-min"] = xcorr_config.getfloat("zmin")
    xcorr_dict["z-cut-max"] = xcorr_config.getfloat("zmax")
    xcorr_dict["z-min-obj"] = xcorr_config.getfloat("zmin")
    xcorr_dict["z-max-obj"] = xcorr_config.getfloat("zmax")

    dict_picca_rmu.update(xcorr_dict)
    dict_picca_rmu["out"] = f"{xcorr_name}.fits"
    dict_picca_rprt.update(xcorr_dict)
    dict_picca_rprt["out"] = f"{xcorr_name}_rprt.fits"

    xcorr.run_cross_corr_picca(dict_picca_rmu,rmu=True)
    xcorr.run_export_picca(f"{xcorr_name}.fits",f"{xcorr_name}_exp.fits",smooth=False)

    xcorr.run_cross_corr_picca(dict_picca_rprt,rmu=False)
    xcorr.run_export_picca(f"{xcorr_name}_rprt.fits",f"{xcorr_name}_rprt_exp.fits",smooth=True)
