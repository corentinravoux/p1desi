"""Shared constants and unit-conversion helpers for the P1D analysis.

Defines the Lyman-alpha rest wavelength, the speed of light and the Si IV / C IV
doublet wavelengths (used by the metal-contamination model), plus helpers to
convert wavenumbers between wavelength units (Angstrom^-1) and velocity units
(s/km) and to add secondary velocity-unit axes to matplotlib figures.
"""

from functools import partial

import numpy as np
from matplotlib.ticker import FuncFormatter

lambdaLy = 1215.673123130217

speed_light = 2.99792458 * 10**5

si4_1 = 1393.76
si4_2 = 1402.77
c4_1 = 1548.202
c4_2 = 1550.774

vsi4 = speed_light * np.log(si4_2 / si4_1)
lsi4 = si4_2 - si4_1
ksi4 = 2 * np.pi / (si4_2 - si4_1)
ksi4_speed = 2 * np.pi / vsi4


vc4 = speed_light * np.log(c4_2 / c4_1)
lc4 = c4_2 - c4_1
kc4 = 2 * np.pi / (c4_2 - c4_1)
kc4_speed = 2 * np.pi / vc4


def return_key(dictionary, string, default_value):
    """Return ``dictionary[string]`` if present, else ``default_value``.

    Args:
        dictionary (dict): Source mapping.
        string: Key to look up.
        default_value: Value returned when the key is absent.

    Returns:
        The value at ``string`` or ``default_value``.
    """
    return dictionary[string] if string in dictionary.keys() else default_value


def return_conversion_factor(z):
    """Wavenumber conversion factor between Angstrom^-1 and s/km at redshift ``z``.

    Args:
        z (float): Redshift.

    Returns:
        float: ``c / (1 + z) / lambda_Lya`` (km/s per Angstrom), i.e. the factor
        mapping ``k`` in s/km to ``k`` in Angstrom^-1.
    """
    return speed_light / (1 + z) / lambdaLy


def kAAtokskm(x, z=2.2):
    """Convert a wavenumber from Angstrom^-1 to s/km.

    Args:
        x (float | array-like): Wavenumber in Angstrom^-1.
        z (float, optional): Redshift. Defaults to 2.2.

    Returns:
        float | array-like: Wavenumber in s/km.
    """
    return x / return_conversion_factor(z)


def kskmtokAA(x, z=2.2):
    """Convert a wavenumber from s/km to Angstrom^-1.

    Args:
        x (float | array-like): Wavenumber in s/km.
        z (float, optional): Redshift. Defaults to 2.2.

    Returns:
        float | array-like: Wavenumber in Angstrom^-1.
    """
    return x * return_conversion_factor(z)


def kAAtokskm_label(x, pos=None, z=2.2):
    """Matplotlib tick formatter turning an Angstrom^-1 tick into an s/km label.

    Args:
        x (float): Tick value in Angstrom^-1.
        pos: Tick position (unused; required by matplotlib).
        z (float, optional): Redshift. Defaults to 2.2.

    Returns:
        str: The wavenumber in s/km, formatted to three decimals.
    """
    kstr = x
    knew = float(kstr) / (speed_light / (1 + z) / lambdaLy)
    transformed_label = "{:.3f}".format(knew)
    return transformed_label


def kskmtokAA_label(x, pos=None, z=2.2):
    """Matplotlib tick formatter turning an s/km tick into an Angstrom^-1 label.

    Args:
        x (float): Tick value in s/km.
        pos: Tick position (unused; required by matplotlib).
        z (float, optional): Redshift. Defaults to 2.2.

    Returns:
        str: The wavenumber in Angstrom^-1, formatted to three decimals.
    """
    kstr = x
    knew = float(kstr) * (speed_light / (1 + z) / lambdaLy)
    transformed_label = "{:.3f}".format(knew)
    return transformed_label


def place_k_speed_unit_axis(fig, ax, fontsize=None, size=None, pos=0.2):
    """Add three secondary top axes showing k in s/km at z = 2.2, 3.0, 3.8.

    Args:
        fig (matplotlib.figure.Figure): Parent figure.
        ax (matplotlib.axes.Axes): Axis (with k in Angstrom^-1) to annotate.
        fontsize (float, optional): Axis-label font size.
        size (float, optional): Tick-label font size.
        pos (float, optional): Vertical offset between the stacked axes.
    """
    # this createss more x-axes to compare things in k[s/km]
    par1 = ax.twiny()
    par2 = ax.twiny()
    par3 = ax.twiny()
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["top"].set_position(("axes", 1 + 1 * pos))
    par3.spines["top"].set_position(("axes", 1 + 2 * pos))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["top"].set_visible(True)
    par1.set_xlabel(r"$k~[$s$\cdot$km$^{-1}]$ @ z=2.2", fontsize=fontsize)
    par2.set_xlabel(r"$k~[$s$\cdot$km$^{-1}]$ @ z=3.0", fontsize=fontsize)
    par3.set_xlabel(r"$k~[$s$\cdot$km$^{-1}]$ @ z=3.8", fontsize=fontsize)

    if size is not None:
        par1.xaxis.set_tick_params(labelsize=size)
        par2.xaxis.set_tick_params(labelsize=size)
        par3.xaxis.set_tick_params(labelsize=size)

    par1.set_xlim(*ax.get_xlim())
    par2.set_xlim(*ax.get_xlim())
    par3.set_xlim(*ax.get_xlim())

    par1.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm_label, z=2.2)))
    par2.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm_label, z=3.0)))
    par3.xaxis.set_major_formatter(FuncFormatter(partial(kAAtokskm_label, z=3.8)))


def place_k_wavelength_unit_axis(fig, ax, z, fontt=None):
    """Add a secondary top axis showing k in s/km for an Angstrom^-1 axis.

    Args:
        fig (matplotlib.figure.Figure): Parent figure.
        ax (matplotlib.axes.Axes): Axis (with k in Angstrom^-1) to annotate.
        z (float): Redshift used for the unit conversion.
        fontt (float, optional): Axis-label font size.
    """
    # this createss more x-axes to compare things in k[s/km]
    par1 = ax.twiny()
    par1.set_xlabel(r" k [s/km] @ z=2.2", fontsize=fontt)
    par1.xaxis.set_major_formatter(FuncFormatter(partial(kskmtokAA_label, z)))


def make_patch_spines_invisible(ax):
    """Make an axis frame transparent, keeping only its spines available.

    Args:
        ax (matplotlib.axes.Axes): Axis to make transparent.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
