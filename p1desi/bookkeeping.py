"""Path and naming conventions for the P1D bookkeeping layout.

Encodes how analysis products (deltas, P1D measurements) are laid out on disk
as a function of the analysis variant (region, line list, catalog, DLA/BAL
masking, noise estimate) and builds the corresponding folder / file names.
"""

import itertools
import os


def get_path(file_name):
    """Read a single path stored in a one-line text file.

    Args:
        file_name (str): Text file holding the path.

    Returns:
        str: The stripped file contents.
    """
    return open(file_name, "r").read().strip()


def get_params(file_name):
    """Read the five analysis parameters from a bookkeeping parameter file.

    Args:
        file_name (str): Text file with one value per line.

    Returns:
        tuple[str, str, str, str, str]: ``(noise_estimate, catalog_name,
        lines_name, dla_name, bal)``.
    """
    f = tuple(open(file_name, "r").read().strip().split("\n"))
    noise_estimate, catalog_name, lines_name, dla_name, bal = (
        f[0],
        f[1],
        f[2],
        f[3],
        f[4],
    )
    return noise_estimate, catalog_name, lines_name, dla_name, bal


def return_folder_name(
    region, lines_name, catalog_name, dla_name, bal, noise_estimate, suffix
):
    """Build the analysis folder name encoding the full variant.

    Args:
        region (str): Spectral region (e.g. ``"lya"``, ``"SB1"``).
        lines_name (str): Sky-line mask identifier.
        catalog_name (str): QSO catalog identifier.
        dla_name (str): DLA catalog / masking identifier.
        bal (str | float): BAL masking threshold identifier.
        noise_estimate (str): Noise-estimate method identifier.
        suffix (str | None): Optional extra suffix (``"None"``/None -> empty).

    Returns:
        str: The composed folder name.
    """
    if (suffix == "None") | (suffix == None):
        suffix = ""
    folder_name = f"{region}_noise{noise_estimate}_lines{lines_name}_cat{catalog_name}_dla{dla_name}_bal{bal}{suffix}"
    return folder_name


def return_delta_path(pk_path, folder_name):
    """Return the deltas directory for a given analysis folder.

    Args:
        pk_path (str): Base analysis directory.
        folder_name (str): Analysis folder name (see :func:`return_folder_name`).

    Returns:
        str: ``{pk_path}/p1d_{folder_name}/deltas``.
    """
    pk = os.path.join(pk_path, f"p1d_{folder_name}", "deltas")
    return pk


def return_pk_path(pk_path, folder_name, snr_cut_path):
    """Return the P1D directory for a given analysis folder and SNR cut.

    Args:
        pk_path (str): Base analysis directory.
        folder_name (str): Analysis folder name (see :func:`return_folder_name`).
        snr_cut_path (int | str): SNR-cut identifier embedded in the path.

    Returns:
        str: ``{pk_path}/p1d_{folder_name}/pk1d_SNRcut{snr_cut_path}``.
    """
    pk = os.path.join(pk_path, f"p1d_{folder_name}", f"pk1d_SNRcut{snr_cut_path}")
    return pk


def return_pk_path_interface(main_config):
    """Resolve all P1D input paths from an interface config.

    When ``use_bookkeeping`` is set, iterates the Cartesian product of the
    configured analysis variants and builds the P1D paths (plus optional
    side-band paths); otherwise reads explicit absolute paths from the config.

    Args:
        main_config: Config object exposing typed getters (``getstr``,
            ``getboolean``, ``getliststr``, ``getlistfloat``, ``getlistint``).

    Returns:
        tuple[list, list, list]: ``(pk, pk_sb, outname)`` — P1D paths, side-band
        P1D paths (or None entries) and output names.
    """
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
    """Build the file name of a mean-P1D product from its options.

    Args:
        velunits (bool): Whether the P1D is in velocity units (s/km).
        logsample (bool): Whether the k binning is logarithmic.
        snr_cut_mean (optional): If set, marks an SNR-cut mean product.

    Returns:
        str: The ``mean_Pk1d*.fits.gz`` file name.
    """
    return f'mean_Pk1d{"_log" if logsample else ""}{"_vel" if velunits else ""}{"_snr_cut_mean" if snr_cut_mean is not None else ""}.fits.gz'
