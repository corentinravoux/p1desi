from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from desispec.io import read_sky
from scipy.ndimage.filters import gaussian_filter
import fitsio, glob, os

lambdaLy = 1215.673123130217


def get_exposure_list(
    path,
    path_exposures,
    survey,
    program,
    exptime_min=None,
    diff_time_max=None,
    seeing_max=None,
    airmass_max=None,
    ebv_max=None,
):
    mask_survey = fitsio.FITS(path)[1]["FAPRGRM"][:] == program
    mask_survey &= fitsio.FITS(path)[1]["SURVEY"][:] == survey
    mask_total = mask_survey
    if exptime_min is not None:
        mask_total &= fitsio.FITS(path)[1]["EXPTIME"][:] > exptime_min
    if diff_time_max is not None:
        mask_total &= (
            fitsio.FITS(path)[1]["EXPTIME"][:] - fitsio.FITS(path)[1]["EFFTIME_SPEC"][:]
            < diff_time_max
        )
    if seeing_max is not None:
        mask_total &= fitsio.FITS(path)[1]["SEEING_ETC"][:] < seeing_max
    if airmass_max is not None:
        mask_total &= fitsio.FITS(path)[1]["AIRMASS"][:] < airmass_max
    if ebv_max is not None:
        mask_total &= fitsio.FITS(path)[1]["EBV"][:] < ebv_max

    print("PATH:")
    print(path)
    print("EXPTIME:")
    print(fitsio.FITS(path)[1]["EXPTIME"][:][mask_total])
    print("EFFTIME:")
    print(fitsio.FITS(path)[1]["EFFTIME_SPEC"][:][mask_total])
    print("EXPTIME - EFFTIME:")
    print(
        fitsio.FITS(path)[1]["EXPTIME"][:][mask_total]
        - fitsio.FITS(path)[1]["EFFTIME_SPEC"][:][mask_total]
    )
    print("SEEING:")
    print(fitsio.FITS(path)[1]["SEEING_ETC"][:][mask_total])
    print("AIRMASS:")
    print(fitsio.FITS(path)[1]["AIRMASS"][:][mask_total])
    print("EBV:")
    print(fitsio.FITS(path)[1]["EBV"][:][mask_total])
    exposures = fitsio.FITS(path)[1]["EXPID"][:][mask_total]
    nights = fitsio.FITS(path)[1]["NIGHT"][:][mask_total]
    list_path = []
    for i in range(len(nights)):
        list_exp = glob.glob(
            os.path.join(
                path_exposures, str(nights[i]), f"*{exposures[i]}", "sky*.fits"
            )
        )
        list_path = list_path + list_exp

    print("EXPID:")
    print(fitsio.FITS(path)[1]["EXPID"][:][mask_total])
    print("NIGHT:")
    print(fitsio.FITS(path)[1]["NIGHT"][:][mask_total])
    return list_path


def get_sky_flux(file_name):
    sky_flux_b, sky_flux_r, sky_flux_z = [], [], []
    for j in range(len(file_name)):
        sky = read_sky(file_name[j])
        if "-b" in file_name[j]:
            sky_flux_b.append(sky.flux)
            sky_wave_b = sky.wave
        if "-r" in file_name[j]:
            sky_flux_r.append(sky.flux)
            sky_wave_r = sky.wave
        if "-z" in file_name[j]:
            sky_flux_z.append(sky.flux)
            sky_wave_z = sky.wave
    sky_flux_b = np.concatenate(sky_flux_b, axis=0)
    sky_flux_r = np.concatenate(sky_flux_r, axis=0)
    sky_flux_z = np.concatenate(sky_flux_z, axis=0)
    return (sky_flux_b, sky_flux_r, sky_flux_z, sky_wave_b, sky_wave_r, sky_wave_z)


def compute_line_width(
    line,
    method_width,
    width_angstrom=1,
    threshold_width=1.2,
    mean_flux=None,
    median_sky=None,
    wave=None,
):

    if method_width.lower() == "constant":
        line[0] = line[0] - width_angstrom
        line[1] = line[1] + width_angstrom

    elif method_width.lower() == "threshold":
        mask_width = mean_flux < threshold_width * median_sky
        if len(wave[mask_width & (wave <= line[0])]) == 0:
            line[0] = line[0] - width_angstrom
        else:
            line[0] = np.max(wave[mask_width & (wave <= line[0])]) - 0.1
        if len(wave[mask_width & (wave >= line[1])]) == 0:
            line[1] = line[1] + width_angstrom
        else:
            line[1] = np.min(wave[mask_width & (wave >= line[1])]) + 0.1
    return line


def add_peak_position(line, mean_flux, median_sky, wavelength):
    mask_peak = (wavelength >= line[0]) & (wavelength <= line[1])
    arg = np.argmax(mean_flux[mask_peak])
    wavelength_peak = wavelength[mask_peak][arg]
    significance_peak = mean_flux[mask_peak][arg] / median_sky[mask_peak][arg]
    return [line[0], line[1], wavelength_peak, significance_peak]


def get_lines_from_sky_file(
    sky_flux_b,
    sky_flux_r,
    sky_flux_z,
    sky_wave_b,
    sky_wave_r,
    sky_wave_z,
    threshold,
    method_width,
    median_size,
    threshold_width,
    width_angstrom=1,
):
    lines = []
    for band in ["b", "r", "z"]:
        thres = threshold[band]
        thres_width = threshold_width[band]
        flux = eval(f"sky_flux_{band}")
        wavelength = eval(f"sky_wave_{band}")
        mean_flux = np.mean(flux, axis=0)
        median_sky = median_filter(mean_flux, median_size)

        ### CR - Other methods to compute median flux. Same results without scipy:
        # median_sky = np.zeros(mean_flux.shape)
        # size = median_size//2
        # median_sky[size:-size] = np.array([np.median(mean_flux[i-size:i+size]) for i in range(size,len(mean_flux)-size)])
        # median_sky[0:size] = np.array([np.median(mean_flux[0:(size+i)]) for i in range(0,size)])
        # median_sky[-size:] = np.array([np.median(mean_flux[i+1-size:len(mean_flux)]) for i in range(len(mean_flux)-size,len(mean_flux))])

        ### CR - Other methods to compute median flux. Median then mean:
        # median_sky = np.mean(np.array([median_filter(flux,median_size) for j in range(len(flux))]),axis=0)
        mask = mean_flux > thres * median_sky
        wavelength_above = wavelength[mask]
        diff = wavelength_above[1:] - wavelength_above[:-1]
        arg_peak = np.argwhere(diff > 1)
        if len(arg_peak) == 0:
            if len(wavelength_above) != 0:
                line = compute_line_width(
                    [np.min(wavelength_above), np.max(wavelength_above)],
                    method_width,
                    width_angstrom=width_angstrom,
                    threshold_width=thres_width,
                    mean_flux=mean_flux,
                    median_sky=median_sky,
                    wave=wavelength,
                )
                line = add_peak_position(line, mean_flux, median_sky, wavelength)
                lines.append(line)
        else:
            arg_init_peak = 0
            for i in range(len(arg_peak)):
                line = compute_line_width(
                    [wavelength_above[arg_init_peak], wavelength_above[arg_peak[i][0]]],
                    method_width,
                    width_angstrom=width_angstrom,
                    threshold_width=thres_width,
                    mean_flux=mean_flux,
                    median_sky=median_sky,
                    wave=wavelength,
                )
                line = add_peak_position(line, mean_flux, median_sky, wavelength)
                lines.append(line)
                arg_init_peak = arg_peak[i][0] + 1
            line = compute_line_width(
                [wavelength_above[arg_peak[-1][0] + 1], wavelength_above[-1]],
                method_width,
                width_angstrom=width_angstrom,
                threshold_width=thres_width,
                mean_flux=mean_flux,
                median_sky=median_sky,
                wave=wavelength,
            )
            line = add_peak_position(line, mean_flux, median_sky, wavelength)
            lines.append(line)
    return np.array(lines)


def merge_overlaping_line(lines):
    lines = np.array(sorted(lines, key=lambda a: a[0], reverse=False))
    starts = lines[:, 0]
    ends = np.maximum.accumulate(lines[:, 1])
    valid = np.zeros(len(lines) + 1, dtype=np.bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = starts[1:] >= ends[:-1]
    arg_valid = np.argwhere(valid)[:, 0]
    arg_notvalid = np.argwhere(~valid)[:, 0]
    arg_merge = []
    for i in range(len(arg_valid)):
        mask_arg_not_valid = arg_notvalid > arg_valid[i]
        if i != len(arg_valid) - 1:
            mask_arg_not_valid &= arg_notvalid < arg_valid[i + 1]
        arg_merge.append([arg_valid[i]] + list(arg_notvalid[mask_arg_not_valid]))
    wavelength_peak, significance_peak = [], []
    for i in range(len(arg_merge) - 1):
        arg_max_significance = np.nanargmax(lines[:, 3][arg_merge[i]])
        significance_peak.append(lines[:, 3][arg_merge[i][arg_max_significance]])
        wavelength_peak.append(lines[:, 2][arg_merge[i][arg_max_significance]])
    return np.vstack(
        (starts[:][valid[:-1]], ends[:][valid[1:]], wavelength_peak, significance_peak)
    ).T


def add_custom_line(lines, custom_lines, names, types):
    list_type = []
    list_name = []
    lines_out = []
    lines_out = []
    custom_lines_mask = np.full(len(custom_lines), True)
    for i in range(len(lines)):
        custom_lines_left = np.argwhere(custom_lines_mask)
        for arg in custom_lines_left:
            if custom_lines[arg[0]][1] < lines[i][0]:
                list_type.append(custom_lines[arg[0]][3])
                list_name.append(custom_lines[arg[0]][0])
                lines_out.append([custom_lines[arg[0]][1], custom_lines[arg[0]][2]])
                custom_lines_mask[arg[0]] = False
        list_type.append(types)
        list_name.append(names)
        lines_out.append(lines[i])
    return (lines_out, list_name, list_type)


def delete_line_number(i, names, lines, types):
    del names[i]
    del lines[i]
    del types[i]


def replace_name(center_line, new_name, names, lines, types):
    for i in range(len(lines)):
        if (lines[i][0] < center_line) & (lines[i][1] > center_line):
            names[i] = new_name
    return (names, lines, types)


def read_lines(lines_file, p1d=False):
    file = open(lines_file, "r")
    file_lines = file.readlines()
    file.close()
    lines, names, types, wave_peak, significance_peak = [], [], [], [], []
    for i in range(len(file_lines)):
        line = file_lines[i].strip()
        if line[0] != "#":
            line = line.split()
            lines.append([float(line[1]), float(line[2])])
            names.append(line[0])
            types.append(line[3])
            if p1d == False:
                wave_peak.append(float(line[4]))
                significance_peak.append(float(line[5]))
    return (np.array(lines), names, types, wave_peak, significance_peak)


def write_skyline_file(name_out, names, lines, types, p1d=False):
    if type(names) != list:
        names = [names for i in range(len(lines))]
    if type(types) != list:
        types = [types for i in range(len(lines))]
    file = open(name_out, "w")
    header = f"""#
# File to veto lines either in observed or rest frame
#       or even rest frame of DLA
#
#  Veto lines used for Pk1D analysis (DESI SV)
#
# lambda is given in Angstrom
#
# name  lambda_min  lambda_max  ('OBS' or 'RF' or 'RF_DLA') {'' if p1d else 'lambda_peak significance_peak'}
#\n"""
    file.write(header)
    for i in range(len(lines)):
        if p1d:
            file.write(
                f"{names[i]}  {np.round(lines[i][0],3)}  {np.round(lines[i][1],3)}  {types[i]}\n"
            )
        else:
            file.write(
                f"{names[i]}  {np.round(lines[i][0],3)}  {np.round(lines[i][1],3)}  {types[i]} {np.round(lines[i][2],3)} {np.round(lines[i][3],3)}\n"
            )
    file.close()


def plot_skylines_on_sky_fiber(
    sky_flux_b,
    sky_flux_r,
    sky_flux_z,
    sky_wave_b,
    sky_wave_r,
    sky_wave_z,
    line_file,
    name_out,
    threshold=None,
    ylim=[0, 500],
    median_size=40,
):
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for band in ["b", "r", "z"]:
        flux = eval(f"sky_flux_{band}")
        wave = eval(f"sky_wave_{band}")
        mean_flux = np.mean(flux, axis=0)
        median_sky = median_filter(mean_flux, median_size)
        ax[0].plot(wave, mean_flux)
        if threshold is not None:
            ax[0].plot(wave, threshold[band] * median_sky, color="r", ls="--")
        ax[1].plot(wave, median_sky)

    legend_elements = [
        Line2D([0], [0], color="C0", lw=1, label="B"),
        Line2D([0], [0], color="C1", lw=1, label="R"),
        Line2D([0], [0], color="C2", lw=1, label="Z"),
    ]
    if threshold is not None:
        legend_elements.append(
            Line2D([0], [0], color="r", lw=1, ls="--", label="Threshold")
        )
    ax[0].legend(handles=legend_elements)

    (line_to_plot, names, types, wave_peak, significance_peak) = read_lines(line_file)
    for i in range(len(line_to_plot)):
        x = np.linspace(line_to_plot[i][0], line_to_plot[i][1], 100)
        for j in [0, 1]:
            ax[j].fill_between(x, ylim[0], ylim[1], color=f"C{1}", alpha=0.3)

    ax[0].set_ylabel("Mean sky fiber flux")
    ax[1].set_ylabel("Median filtered mean sky fiber flux")
    ax[1].set_xlabel("Wavelength")

    ax[0].set_ylim([ylim[0], ylim[1]])
    ax[1].set_ylim([ylim[0], ylim[1]])
    plt.savefig(f"{name_out}_sky_fiber.pdf", format="pdf")


### Plots on stacked noise


def plot_skyline_analysis(
    name_out, flux, wavelength, lines, outlier_insensitive=False, gaussian_smoothing=40
):
    if outlier_insensitive:
        mean_spectra = np.nanmedian(flux, axis=0)
    else:
        mean_spectra = np.nanmean(flux, axis=0)

    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(wavelength, mean_spectra)
    ax[0].set_ylabel("Mean spectra")

    # mean_median =np.mean(np.array([median_filter(flux[i],median_smoothing) for i in range(len(flux))]),axis=0)
    # mean_median =median_filter(mean_spectra,median_smoothing)

    gaussian_mean = gaussian_filter(mean_spectra, gaussian_smoothing)

    ax[1].plot(wavelength, gaussian_mean, color="b")
    ax[1].set_ylabel("Mean spectra smoothed")

    ratio = np.abs((gaussian_mean - mean_spectra) / gaussian_mean)
    ax[2].plot(wavelength, ratio, color="b")
    ax[3].plot(wavelength, ratio, color="b")
    ax[2].set_ylabel("Ratio + eBOSS lines")
    ax[3].set_ylabel("Ratio + DESI lines")
    j = 1
    for line in lines:
        j = j + 1
        (line_to_plot, names, types, wave_peak, significance_peak) = read_lines(
            line, p1d=True
        )
        for i in range(len(line_to_plot)):
            ax[j].fill_between(
                wavelength,
                np.min(ratio),
                np.max(ratio),
                where=(
                    (wavelength > line_to_plot[i][0])
                    & (wavelength < line_to_plot[i][1])
                ),
                color=f"C{1}",
                alpha=0.3,
            )

    fig.savefig(f"{name_out}_mean_spectra.pdf", format="pdf")


def plot_centered_lines(
    name_out,
    array_list,
    wavelength,
    lines,
    outlier_insensitive=False,
    diff_lambda=10,
    title="Mean flux",
):
    if outlier_insensitive:
        mean_spectra = np.nanmedian(array_list, axis=0)
    else:
        mean_spectra = np.nanmean(array_list, axis=0)

    (line_to_plot, names, types, wave_peak, significance_peak) = read_lines(
        lines, p1d=True
    )
    for i in range(len(line_to_plot)):

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(8, 8),
            gridspec_kw={"height_ratios": [3, 1], "width_ratios": [3, 1]},
            sharex=False,
        )
        mean_line = (line_to_plot[i][0] + line_to_plot[i][1]) / 2

        mask = wavelength >= mean_line - diff_lambda
        mask &= wavelength <= mean_line + diff_lambda

        ax[0][0].plot(wavelength[mask], mean_spectra[mask], "x-")
        ax[0][0].grid()
        ax[0][0].set_title(title)
        x = np.linspace(line_to_plot[i][0], line_to_plot[i][1], 100)
        ax[0][0].fill_between(
            x,
            np.min(mean_spectra[mask]),
            np.max(mean_spectra[mask]),
            color=f"C{1}",
            alpha=0.3,
        )

        ax[0][1].plot(wavelength, mean_spectra)
        x = np.linspace(line_to_plot[i][0], line_to_plot[i][1], 100)
        ax[0][1].fill_between(
            x, np.min(mean_spectra), np.max(mean_spectra), color=f"C{1}", alpha=0.3
        )
        ax[0][1].set_xlim([mean_line - diff_lambda, mean_line + diff_lambda])

        ax[1][0].plot(wavelength, mean_spectra)
        ax[1][0].fill_between(
            x, np.min(mean_spectra), np.max(mean_spectra), color=f"C{1}", alpha=0.3
        )
        ax[1][0].set_xlabel("Observed wavelength")
        fig.delaxes(ax[1][1])
        fig.savefig(
            f"{name_out}_line_centered_{i:03d}_{np.around(mean_line,0)}.png",
            format="png",
        )
        plt.close()


def compute_length_masked(lines, redshift_bins):
    percentage_mask = []
    (line_to_plot, names, types, wave_peak, significance_peak) = read_lines(
        lines, p1d=True
    )
    for i in range(len(redshift_bins)):
        lambda_min = (1 + redshift_bins[i] - 0.1) * lambdaLy
        lambda_max = (1 + redshift_bins[i] + 0.1) * lambdaLy
        sum = 0
        for j in range(len(line_to_plot)):
            if (line_to_plot[j][1] < lambda_max) & (line_to_plot[j][0] > lambda_min):
                sum = sum + line_to_plot[j][1] - line_to_plot[j][0]
        percentage_mask.append(100 * sum / (lambda_max - lambda_min))
    return percentage_mask


def plot_percentage_mask(redshift_bins, percentage_mask, legend, nameout):
    plt.figure()
    plt.ylabel("Percentage masked")
    for i in range(len(percentage_mask)):
        plt.plot(redshift_bins, percentage_mask[i])
    plt.legend(legend)
    plt.savefig(f"{nameout}.pdf", format="pdf")
