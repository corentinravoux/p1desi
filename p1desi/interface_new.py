import os
import configparser
import ast
import numpy as np
from p1desi import resolution, bookkeeping


def parse_int_tuple(input):
    if input == "None":
        return None
    else:
        return tuple(int(k.strip()) for k in input.strip().split(","))


def parse_float_tuple(input):
    if input == "None":
        return None
    else:
        return tuple(float(k.strip()) for k in input.strip().split(","))


def parse_str_tuple(input):
    if input == "None":
        return None
    else:
        return tuple(str(k.strip()) for k in input.strip().split(","))


def parse_int_list(input):
    if input == "None":
        return [None]
    else:
        return list(int(k.strip()) for k in input.strip().split(","))


def parse_float_list(input):
    if input == "None":
        return [None]
    else:
        return list(float(k.strip()) for k in input.strip().split(","))


def parse_str_list(input):
    if input == "None":
        return [None]
    else:
        return list(str(k.strip()) for k in input.strip().split(","))


def parse_dict(input):
    if input == "None":
        return None
    else:
        acceptable_string = input.replace("'", '"')
        return ast.literal_eval(acceptable_string)


def parse_float(input):
    if input == "None":
        return None
    else:
        return float(input)


def parse_int(input):
    if input == "None":
        return None
    else:
        return int(input)


def parse_string(input):
    if input == "None":
        return None
    else:
        return str(input)


def main(input_file):
    config = configparser.ConfigParser(
        allow_no_value=True,
        converters={
            "str": parse_string,
            "int": parse_int,
            "float": parse_float,
            "tupleint": parse_int_tuple,
            "tuplefloat": parse_float_tuple,
            "tuplestr": parse_str_tuple,
            "listint": parse_int_list,
            "listfloat": parse_float_list,
            "liststr": parse_str_list,
            "dict": parse_dict,
        },
    )
    config.optionxform = lambda option: option
    config.read(input_file)

    main_config = config["main"]
    path_config = config["path"]
    main_path = os.path.abspath(path_config["path"])
    os.makedirs(main_path, exist_ok=True)

    (pks, pk_sbs, outnames) = bookkeeping.return_pk_path_interface(path_config)

    for pk, pk_sb, outname in zip(pks, pk_sbs, outnames):

        if main_config.getboolean("plot_resolution"):
            config_plot_resolution = config["plot resolution"]
            path_plot_resolution = os.path.join(
                main_path, config_plot_resolution.getstr("path_plot_resolution")
            )
            os.makedirs(path_plot_resolution, exist_ok=True)
            print("Plotting resolution for path: ", pk)
            plot_resolution(
                pk, path_plot_resolution, config_plot_resolution, main_config, outname
            )

        # if main_config.getboolean("plot_noise"):
        #     plot_noise_config = config["plot noise"]
        #     path_plot_noise = os.path.join(
        #         main_path, plot_noise_config.getstr("path_plot_noise")
        #     )
        #     os.makedirs(path_plot_noise, exist_ok=True)
        #     print("Plotting noise study for path: ", pk)
        #     plot_noise(
        #         mean_pk, path_plot_noise, plot_noise_config, main_config, outname
        #     )


def plot_resolution(
    pk, path_plot_resolution, config_plot_resolution, main_config, outname
):
    outname = os.path.join(path_plot_resolution, outname)

    resolution.plot_mean_resolution(
        pk,
        config_plot_resolution.getfloat("zmax"),
        f"{outname}_mean_resolution.pdf",
        f"{outname}_mean_resolution.txt",
        kmax_line=config_plot_resolution.getfloat("kmax_line"),
        **config_plot_resolution.getdict("plot_args_resolution"),
    )
    resolution.fit_resolution_redshift(
        pk,
        config_plot_resolution.getfloat("zmax"),
        f"{outname}_resolution_fit_points.pickle",
    )


# def plot_noise(mean_pk, path_plot_noise, plot_noise_config, main_config, outname):
#     outname = os.path.join(path_plot_noise, outname)
#     data = plotpk.read_pk_means(mean_pk)

#     plotpk.plot_noise_study(
#         data,
#         np.array(main_config.gettuplefloat("zbins_plot")),
#         outname,
#         plot_noise_config.getstr("k_units_noise_study"),
#         plot_noise_config.getboolean("use_diff_noise"),
#         plot_noise_config.getboolean("plot_noise_ratio"),
#         plot_noise_config.getboolean("plot_noise_comparison_mean_k"),
#         plot_noise_config.getboolean("plot_side_band"),
#         side_band_comp=plot_noise_config.getstr("side_band_comp"),
#         side_band_legend=plot_noise_config.gettuplestr("side_band_legend"),
#         fit_asymptote_ratio=plot_noise_config.getboolean("fit_asymptote_ratio"),
#         plot_difference=plot_noise_config.getboolean("plot_difference"),
#         **plot_noise_config.getdict("plot_args_noise"),
#     )

#     if plot_noise_config.getboolean("plot_noise_comparison_mean_z"):
#         noise.compute_and_plot_mean_z_noise_power(
#             data,
#             np.array(main_config.gettuplefloat("zbins_plot")),
#             outname,
#             **plot_noise_config.getdict("plot_args_noise"),
#         )
