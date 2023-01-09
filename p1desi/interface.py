import os
import configparser
import ast
import numpy as np
from p1desi import plotpk, bookkeeping, noise


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
        velunits = main_config.getboolean("velunits")
        logsample = main_config.getboolean("logsample")
        snr_cut_mean = main_config.gettuplefloat("snr_cut_mean")
        mean_pk = os.path.join(
            pk,
            bookkeeping.return_mean_pk_name(
                velunits, logsample, snr_cut_mean=snr_cut_mean
            ),
        )
        if pk_sb is not None:
            mean_pk_sb = os.path.join(
                pk_sb,
                bookkeeping.return_mean_pk_name(
                    velunits, logsample, snr_cut_mean=snr_cut_mean
                ),
            )
        else:
            mean_pk_sb = None
        suffix_plot = main_config.getstr("suffix_plot")
        if suffix_plot is not None:
            outname = f"{outname}_{suffix_plot}_unit{'kms' if velunits else 'A'}"
        else:
            outname = f"{outname}_unit{'kms' if velunits else 'A'}"

        if main_config.getboolean("plot_power"):
            plot_config = config["plot power"]
            path_plot = os.path.join(main_path, plot_config.getstr("path_plot"))
            os.makedirs(path_plot, exist_ok=True)
            print("Plotting pk1d for path: ", pk)
            plot(mean_pk, mean_pk_sb, path_plot, plot_config, main_config, outname)

        if main_config.getboolean("plot_noise"):
            plot_noise_config = config["plot noise"]
            path_plot_noise = os.path.join(
                main_path, plot_noise_config.getstr("path_plot_noise")
            )
            os.makedirs(path_plot_noise, exist_ok=True)
            print("Plotting noise study for path: ", pk)
            plot_noise(
                mean_pk, path_plot_noise, plot_noise_config, main_config, outname
            )

        if main_config.getboolean("plot_metals"):
            plot_metals_config = config["plot metals"]
            path_plot_metals = os.path.join(
                main_path, plot_metals_config.getstr("path_plot_metals")
            )
            os.makedirs(path_plot_metals, exist_ok=True)
            print("Plotting metals study for path: ", pk)
            plot_metals(
                mean_pk, path_plot_metals, plot_metals_config, main_config, outname
            )


def plot(mean_pk, mean_pk_sb, path_plot, plot_config, main_config, outname):
    if plot_config.getboolean("substract_sb"):
        substract_data = mean_pk_sb
        outname = f"{outname}_{plot_config.getstr('region_sb')}_substracted"
    else:
        substract_data = None

    comparison_model_file = plot_config.gettuplestr("comparison_model_file")
    if comparison_model_file is not None:
        comparison_model_file = list(plot_config.gettuplestr("comparison_model_file"))
        if len(comparison_model_file) == 1:
            comparison_model_file = comparison_model_file[0]
    outname = os.path.join(
        path_plot, f"{outname}_model{plot_config.getstr('comparison_str')}"
    )

    plotpk.plot_data(
        plotpk.read_pk_means(mean_pk),
        np.array(main_config.gettuplefloat("zbins_plot")),
        outname,
        plot_P=plot_config.getboolean("plot_P"),
        comparison=plot_config.getstr("pk_comparison"),
        comparison_model=plot_config.getstr("model_comparison"),
        comparison_model_file=comparison_model_file,
        plot_diff=plot_config.getboolean("plot_diff_model"),
        substract_sb=substract_data,
        beta_correction=plot_config.getfloat("beta_correction"),
        beta_correction_sb=plot_config.getfloat("beta_correction_sb"),
        **plot_config.getdict("plot_args"),
    )


def plot_noise(mean_pk, path_plot_noise, plot_noise_config, main_config, outname):
    outname = os.path.join(path_plot_noise, outname)
    data = plotpk.read_pk_means(mean_pk)

    plotpk.plot_noise_study(
        data,
        np.array(main_config.gettuplefloat("zbins_plot")),
        outname,
        plot_noise_config.getstr("k_units_noise_study"),
        plot_noise_config.getboolean("use_diff_noise"),
        plot_noise_config.getboolean("plot_noise_ratio"),
        plot_noise_config.getboolean("plot_noise_comparison_mean_k"),
        plot_noise_config.getboolean("plot_side_band"),
        side_band_comp=plot_noise_config.getstr("side_band_comp"),
        side_band_legend=plot_noise_config.gettuplestr("side_band_legend"),
        fit_asymptote_ratio=plot_noise_config.getboolean("fit_asymptote_ratio"),
        plot_difference=plot_noise_config.getboolean("plot_difference"),
        **plot_noise_config.getdict("plot_args_noise"),
    )

    if plot_noise_config.getboolean("plot_noise_comparison_mean_z"):
        noise.compute_and_plot_mean_z_noise_power(
            data,
            np.array(main_config.gettuplefloat("zbins_plot")),
            outname,
            **plot_noise_config.getdict("plot_args_noise"),
        )


def plot_metals(mean_pk, path_plot_metals, plot_metals_config, main_config, outname):
    outname = os.path.join(path_plot_metals, outname)
    data = plotpk.read_pk_means(mean_pk)

    plotpk.plot_metal_study(
        data,
        np.array(main_config.gettuplefloat("zbins_plot")),
        outname,
        plot_metals_config.getstr("k_units_metals_study"),
        plot_metals_config.getboolean("use_diff_noise"),
        plot_metals_config.getboolean("plot_side_band"),
        side_band_comp=plot_metals_config.getstr("side_band_comp"),
        side_band_legend=plot_metals_config.gettuplestr("side_band_legend"),
        side_band_fitpolynome=plot_metals_config.getboolean("side_band_fitpolynome"),
        **plot_metals_config.getdict("plot_args_metals"),
    )
