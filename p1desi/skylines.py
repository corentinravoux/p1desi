"""Sky-line identification and masking utilities for DESI P1D analysis.

This module provides tools to detect residual sky-emission features in
stacked DESI sky-fiber spectra or in stacked forest noise power, to turn
those detections into wavelength-interval masks ("veto lines" in
Angstrom, observed frame), to merge/curate the resulting line lists
(including eBOSS/DESI custom lines), to read and write the ASCII
skyline files consumed by the P1D pipeline, and to produce diagnostic
plots of the detected lines and of the fraction of the forest masked
per redshift bin.
"""

import glob
import os

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from desispec.io import read_sky
from matplotlib.lines import Line2D
from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_filter

from p1desi import utils


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
    """Select exposures matching survey/program and quality cuts, and list their sky files.

    Reads an exposure table FITS file, builds a boolean selection mask from
    the survey/program identifiers and optional quality cuts (minimum
    exposure time, maximum exposure-vs-effective-time difference, maximum
    seeing, maximum airmass, maximum E(B-V)), prints the selected exposure
    diagnostics, and globs the corresponding per-exposure sky FITS files on
    disk.

    Args:
        path (str): Path to the exposure table FITS file (with columns
            FAPRGRM, SURVEY, EXPTIME, EFFTIME_SPEC, SEEING_ETC, AIRMASS,
            EBV, EXPID, NIGHT).
        path_exposures (str): Root directory containing per-night
            per-exposure directories with ``sky*.fits`` files.
        survey (str): Survey name to select (matched against SURVEY column).
        program (str): Program name to select (matched against FAPRGRM
            column).
        exptime_min (float, optional): Minimum EXPTIME (seconds) to keep an
            exposure. Defaults to None (no cut).
        diff_time_max (float, optional): Maximum allowed EXPTIME -
            EFFTIME_SPEC (seconds). Defaults to None (no cut).
        seeing_max (float, optional): Maximum SEEING_ETC. Defaults to None
            (no cut).
        airmass_max (float, optional): Maximum AIRMASS. Defaults to None (no
            cut).
        ebv_max (float, optional): Maximum E(B-V). Defaults to None (no cut).

    Returns:
        list of str: Paths to the ``sky*.fits`` files of all exposures
        passing the selection.
    """
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
    """Read and concatenate sky-fiber flux and wavelength grids per DESI camera arm.

    Loads each file with ``desispec.io.read_sky`` and dispatches it into the
    b, r, or z arm arrays based on the ``-b``/``-r``/``-z`` substring in the
    file name, then concatenates the per-exposure flux arrays along the
    fiber axis for each arm.

    Args:
        file_name (list of str): Paths to per-exposure ``sky*.fits`` files,
            each containing a camera-arm tag (``-b``, ``-r``, or ``-z``) in
            its name.

    Returns:
        tuple: ``(sky_flux_b, sky_flux_r, sky_flux_z, sky_wave_b,
        sky_wave_r, sky_wave_z)`` where the flux arrays are
        ``numpy.ndarray`` of shape (n_fibers_total, n_wave) concatenated
        over all input files for that arm, and the wave arrays are the
        (last-read) 1D wavelength grid in Angstrom for that arm.
    """
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
    """Widen a detected sky-line interval to a final [min, max] wavelength range.

    Two widening strategies are supported: a fixed additive padding
    ("constant"), or a data-driven extension out to where the mean flux
    drops back below a threshold times the local median sky level
    ("threshold").

    Args:
        line (list of float): Two-element ``[lambda_min, lambda_max]`` in
            Angstrom, modified in place and also returned.
        method_width (str): Widening method, ``"constant"`` or
            ``"threshold"`` (case-insensitive).
        width_angstrom (float, optional): Padding in Angstrom added on each
            side for the "constant" method, or used as fallback padding for
            "threshold" when no wavelength satisfies the threshold
            condition. Defaults to 1.
        threshold_width (float, optional): Multiplicative factor applied to
            ``median_sky`` to define the flux threshold used by the
            "threshold" method. Defaults to 1.2.
        mean_flux (numpy.ndarray, optional): Mean sky flux array, required
            for the "threshold" method.
        median_sky (numpy.ndarray, optional): Median-filtered mean flux
            (local continuum estimate), required for the "threshold"
            method.
        wave (numpy.ndarray, optional): Wavelength grid in Angstrom matching
            ``mean_flux``/``median_sky``, required for the "threshold"
            method.

    Returns:
        list of float: The updated ``[lambda_min, lambda_max]`` line
        boundaries in Angstrom.
    """
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
    """Append the peak wavelength and significance of a line to its boundaries.

    Finds the wavelength of maximum mean flux within the ``[line[0],
    line[1]]`` interval and computes its significance as the ratio of the
    peak flux to the local median sky level at that wavelength.

    Args:
        line (list of float): Two-element ``[lambda_min, lambda_max]`` line
            boundaries in Angstrom.
        mean_flux (numpy.ndarray): Mean flux array over the wavelength grid.
        median_sky (numpy.ndarray): Median-filtered flux (local continuum
            estimate) over the same wavelength grid.
        wavelength (numpy.ndarray): Wavelength grid in Angstrom matching
            ``mean_flux``/``median_sky``.

    Returns:
        list: ``[lambda_min, lambda_max, wavelength_peak,
        significance_peak]``, i.e. the input boundaries followed by the
        peak wavelength (Angstrom) and the peak-to-median significance
        ratio.
    """
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
    """Detect sky-emission line intervals from stacked sky-fiber flux in each camera arm.

    For each of the b, r, z arms, computes the mean flux across fibers and
    a median-filtered local continuum, flags wavelengths where the mean
    flux exceeds ``threshold * median_sky``, groups contiguous flagged
    wavelength stretches (gap > 1 Angstrom starts a new group) into
    candidate lines, widens each candidate via
    :func:`compute_line_width`, and attaches its peak position/significance
    via :func:`add_peak_position`.

    Args:
        sky_flux_b (numpy.ndarray): Sky flux array (n_fibers, n_wave) for
            the b arm.
        sky_flux_r (numpy.ndarray): Sky flux array (n_fibers, n_wave) for
            the r arm.
        sky_flux_z (numpy.ndarray): Sky flux array (n_fibers, n_wave) for
            the z arm.
        sky_wave_b (numpy.ndarray): Wavelength grid in Angstrom for the b
            arm.
        sky_wave_r (numpy.ndarray): Wavelength grid in Angstrom for the r
            arm.
        sky_wave_z (numpy.ndarray): Wavelength grid in Angstrom for the z
            arm.
        threshold (dict): Mapping from band name (``"b"``, ``"r"``,
            ``"z"``) to the flux-detection threshold multiplier applied to
            the median sky level.
        method_width (str): Line-widening method passed to
            :func:`compute_line_width` (``"constant"`` or ``"threshold"``).
        median_size (int): Window size (in samples) of the median filter
            used to estimate the local continuum.
        threshold_width (dict): Mapping from band name to the threshold
            multiplier used by :func:`compute_line_width` when
            ``method_width == "threshold"``.
        width_angstrom (float, optional): Padding in Angstrom passed to
            :func:`compute_line_width`. Defaults to 1.

    Returns:
        numpy.ndarray: Array of shape (n_lines, 4) with columns
        ``[lambda_min, lambda_max, wavelength_peak, significance_peak]``
        (Angstrom), concatenated over the three camera arms.
    """
    lines = []
    for band in ["b", "r", "z"]:
        thres = threshold[band]
        thres_width = threshold_width[band]
        flux = eval(f"sky_flux_{band}")
        wavelength = eval(f"sky_wave_{band}")
        mean_flux = np.mean(flux, axis=0)
        median_sky = median_filter(mean_flux, median_size)

        # Other methods to compute median flux. Same results without scipy:
        # median_sky = np.zeros(mean_flux.shape)
        # size = median_size//2
        # median_sky[size:-size] = np.array([np.median(mean_flux[i-size:i+size]) for i in range(size,len(mean_flux)-size)])
        # median_sky[0:size] = np.array([np.median(mean_flux[0:(size+i)]) for i in range(0,size)])
        # median_sky[-size:] = np.array([np.median(mean_flux[i+1-size:len(mean_flux)]) for i in range(len(mean_flux)-size,len(mean_flux))])

        # Other methods to compute median flux. Median then mean:
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
    """Merge overlapping or touching wavelength-line intervals into single intervals.

    Sorts the input lines by their lower boundary, then merges any lines
    whose interval overlaps with the running maximum of previously seen
    upper boundaries. For each merged group, keeps the peak
    wavelength/significance of the sub-line with the highest significance.

    Args:
        lines (numpy.ndarray or list): Array-like of shape (n_lines, 4)
            with columns ``[lambda_min, lambda_max, wavelength_peak,
            significance_peak]`` in Angstrom.

    Returns:
        numpy.ndarray: Array of shape (n_merged_lines, 4) with columns
        ``[lambda_min, lambda_max, wavelength_peak, significance_peak]``
        after merging overlapping intervals.
    """
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
    """Interleave a list of custom (manually specified) lines into a detected-line list.

    Walks through ``lines`` in order and, before each detected line,
    inserts any not-yet-used custom line whose upper boundary lies below
    the current detected line's lower boundary, preserving overall
    wavelength ordering. Remaining custom lines that never get inserted
    (i.e. lying after the last detected line) are dropped.

    Args:
        lines (numpy.ndarray or list): Array-like of shape (n_lines, >=2)
            with at least ``[lambda_min, lambda_max, ...]`` per detected
            line, assumed sorted by ``lambda_min``.
        custom_lines (list): Sequence of custom line entries, each indexable
            as ``custom_lines[k] = [name, lambda_min, lambda_max, type]``.
        names (str): Name to assign to every entry taken from ``lines``
            (the same value is used for all detected lines).
        types (str): Type label (e.g. ``"OBS"``, ``"RF"``, ``"RF_DLA"``) to
            assign to every entry taken from ``lines``.

    Returns:
        tuple: ``(lines_out, list_name, list_type)`` where ``lines_out`` is
        the merged list of line boundaries (custom lines as
        ``[lambda_min, lambda_max]`` interleaved with the original
        ``lines`` entries), ``list_name`` the corresponding names, and
        ``list_type`` the corresponding type labels.
    """
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
    """Delete the entry at index ``i`` from the parallel names/lines/types lists, in place.

    Args:
        i (int): Index of the entry to remove.
        names (list): List of line names, mutated in place.
        lines (list): List of line boundaries, mutated in place.
        types (list): List of line type labels, mutated in place.

    Returns:
        None
    """
    del names[i]
    del lines[i]
    del types[i]


def replace_name(center_line, new_name, names, lines, types):
    """Rename the line whose interval contains a given wavelength.

    Args:
        center_line (float): Wavelength in Angstrom used to locate the
            line: the line whose ``[lambda_min, lambda_max]`` strictly
            brackets this value gets renamed.
        new_name (str): New name to assign to the matching line.
        names (list of str): Line names, mutated in place for the matching
            entry.
        lines (list): List of ``[lambda_min, lambda_max, ...]`` line
            boundaries, unmodified.
        types (list): Line type labels, unmodified.

    Returns:
        tuple: ``(names, lines, types)`` with ``names`` updated in place.
    """
    for i in range(len(lines)):
        if (lines[i][0] < center_line) & (lines[i][1] > center_line):
            names[i] = new_name
    return (names, lines, types)


def read_lines(lines_file, p1d=False):
    """Read a skyline ASCII veto-line file into its constituent arrays.

    Parses a whitespace-separated file where each non-comment line (lines
    starting with ``#`` are skipped) has the format ``name lambda_min
    lambda_max type [wave_peak significance_peak]``.

    Args:
        lines_file (str): Path to the skyline file to read.
        p1d (bool, optional): If True, the file has no
            ``wave_peak``/``significance_peak`` columns (P1D veto-line
            format) and those outputs are returned empty. Defaults to
            False.

    Returns:
        tuple: ``(lines, names, types, wave_peak, significance_peak)``
        where ``lines`` is a ``numpy.ndarray`` of shape (n_lines, 2) with
        ``[lambda_min, lambda_max]`` in Angstrom, ``names`` and ``types``
        are lists of str, and ``wave_peak``/``significance_peak`` are
        lists of float (empty when ``p1d`` is True).
    """
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
    """Write a list of veto lines to a skyline ASCII file readable by :func:`read_lines`.

    Args:
        name_out (str): Output file path.
        names (str or list of str): Line name, or a list of one name per
            line. If a single str is given, it is broadcast to every line.
        lines (list): Sequence of line boundaries; each entry is
            ``[lambda_min, lambda_max]`` (Angstrom) when ``p1d`` is True,
            or ``[lambda_min, lambda_max, wave_peak, significance_peak]``
            when ``p1d`` is False.
        types (str or list of str): Line type (e.g. ``"OBS"``, ``"RF"``,
            ``"RF_DLA"``), or a list of one type per line. If a single str
            is given, it is broadcast to every line.
        p1d (bool, optional): If True, write the P1D veto-line format
            without peak wavelength/significance columns. Defaults to
            False.

    Returns:
        None
    """
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
    """Plot mean and median-filtered sky-fiber flux with detected/veto lines overlaid.

    Produces a two-panel figure (mean flux and median-filtered mean flux
    per camera arm) with the detection threshold overlaid (if given) and
    shaded bands marking the veto-line intervals read from ``line_file``.
    Saves the figure to ``{name_out}_sky_fiber.pdf``.

    Args:
        sky_flux_b (numpy.ndarray): Sky flux array (n_fibers, n_wave) for
            the b arm.
        sky_flux_r (numpy.ndarray): Sky flux array (n_fibers, n_wave) for
            the r arm.
        sky_flux_z (numpy.ndarray): Sky flux array (n_fibers, n_wave) for
            the z arm.
        sky_wave_b (numpy.ndarray): Wavelength grid in Angstrom for the b
            arm.
        sky_wave_r (numpy.ndarray): Wavelength grid in Angstrom for the r
            arm.
        sky_wave_z (numpy.ndarray): Wavelength grid in Angstrom for the z
            arm.
        line_file (str): Path to a skyline file readable by
            :func:`read_lines` (non-P1D format) giving the lines to
            highlight.
        name_out (str): Output file basename; the figure is saved as
            ``{name_out}_sky_fiber.pdf``.
        threshold (dict, optional): Mapping from band name to a detection
            threshold multiplier, plotted as a dashed reference curve on
            top of the median sky level. Defaults to None (not plotted).
        ylim (list of float, optional): ``[ymin, ymax]`` y-axis limits and
            shading extent for both panels. Defaults to [0, 500].
        median_size (int, optional): Window size (in samples) of the
            median filter applied to the mean flux. Defaults to 40.

    Returns:
        None
    """
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
    """Plot stacked spectra, their Gaussian-smoothed continuum, and the fractional residual with veto lines.

    Builds a four-panel figure: the mean (or median) stacked spectrum, its
    Gaussian-smoothed version, and two copies of the fractional residual
    ``|smoothed - mean| / smoothed`` each overlaid with shaded bands for a
    different veto-line file (e.g. eBOSS lines and DESI lines). Saves the
    figure to ``{name_out}_mean_spectra.pdf``.

    Args:
        name_out (str): Output file basename; the figure is saved as
            ``{name_out}_mean_spectra.pdf``.
        flux (numpy.ndarray): Stacked flux (or power) array of shape
            (n_spectra, n_wave).
        wavelength (numpy.ndarray): Wavelength grid in Angstrom matching
            ``flux``.
        lines (list of str): Paths to one or more P1D-format skyline files
            (readable by :func:`read_lines` with ``p1d=True``), one panel
            of shaded bands is drawn per file (up to 2).
        outlier_insensitive (bool, optional): If True, use the nanmedian
            across spectra instead of the nanmean. Defaults to False.
        gaussian_smoothing (float, optional): Standard deviation (in
            samples) of the Gaussian filter used to estimate the smooth
            continuum. Defaults to 40.

    Returns:
        None
    """
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
    """Plot the stacked spectrum zoomed around each veto line, one PNG per line.

    For every line in ``lines`` (P1D format), draws a 2x2 figure showing
    the stacked spectrum zoomed to +/- ``diff_lambda`` Angstrom around the
    line center (top-left), the same zoom without the shaded line band
    (top-right), and the full-range spectrum (bottom-left) with the line
    band shaded, then saves one PNG per line.

    Args:
        name_out (str): Output file basename; each figure is saved as
            ``{name_out}_line_centered_{i:03d}_{wavelength}.png``.
        array_list (numpy.ndarray): Stacked flux (or power) array of shape
            (n_spectra, n_wave).
        wavelength (numpy.ndarray): Wavelength grid in Angstrom matching
            ``array_list``.
        lines (str): Path to a P1D-format skyline file readable by
            :func:`read_lines` with ``p1d=True``.
        outlier_insensitive (bool, optional): If True, use the nanmedian
            across spectra instead of the nanmean. Defaults to False.
        diff_lambda (float, optional): Half-width in Angstrom of the
            zoomed window plotted around each line center. Defaults to 10.
        title (str, optional): Title used for the zoomed top-left panel.
            Defaults to "Mean flux".

    Returns:
        None
    """
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
    """Compute the percentage of the Lyman-alpha forest masked by veto lines, per redshift bin.

    For each redshift bin, defines the forest wavelength window as
    ``(1 + z - 0.1) * lambda_lya`` to ``(1 + z + 0.1) * lambda_lya`` (using
    :data:`p1desi.utils.lambdaLy`) and sums the widths (in Angstrom) of all
    veto lines from ``lines`` that fall entirely within that window,
    expressed as a percentage of the window width.

    Args:
        lines (str): Path to a P1D-format skyline file readable by
            :func:`read_lines` with ``p1d=True``.
        redshift_bins (list of float or numpy.ndarray): Redshift values at
            which the forest window (and masked percentage) is evaluated.

    Returns:
        list of float: Percentage of the forest window masked by veto
        lines, one value per entry in ``redshift_bins``.
    """
    percentage_mask = []
    (line_to_plot, _, _, _, _) = read_lines(lines, p1d=True)
    for i in range(len(redshift_bins)):
        lambda_min = (1 + redshift_bins[i] - 0.1) * utils.lambdaLy
        lambda_max = (1 + redshift_bins[i] + 0.1) * utils.lambdaLy
        sum = 0
        for j in range(len(line_to_plot)):
            if (line_to_plot[j][1] < lambda_max) & (line_to_plot[j][0] > lambda_min):
                sum = sum + line_to_plot[j][1] - line_to_plot[j][0]
        percentage_mask.append(100 * sum / (lambda_max - lambda_min))
    return percentage_mask


def plot_percentage_mask(redshift_bins, percentage_mask, legend, nameout):
    """Plot the percentage of the forest masked by veto lines as a function of redshift.

    Draws one curve per entry in ``percentage_mask`` against
    ``redshift_bins`` and saves the figure to ``{nameout}.pdf``.

    Args:
        redshift_bins (list of float or numpy.ndarray): Redshift bin
            centers (x-axis), shared across all curves.
        percentage_mask (list of list of float): One masked-percentage
            curve per entry (e.g. as returned by
            :func:`compute_length_masked`), each matching
            ``redshift_bins`` in length.
        legend (list of str): Legend labels, one per curve in
            ``percentage_mask``.
        nameout (str): Output file basename; the figure is saved as
            ``{nameout}.pdf``.

    Returns:
        None
    """
    plt.figure()
    plt.ylabel("Percentage masked")
    for i in range(len(percentage_mask)):
        plt.plot(redshift_bins, percentage_mask[i])
    plt.legend(legend)
    plt.savefig(f"{nameout}.pdf", format="pdf")
