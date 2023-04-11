import numpy as np
from matplotlib.ticker import FuncFormatter
from functools import partial

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
    return dictionary[string] if string in dictionary.keys() else default_value


def kAAtokskm(x, z=2.2):
    return x / (speed_light / (1 + z) / lambdaLy)

def kskmtokAA(x, z=2.2):
    return x * (speed_light / (1 + z) / lambdaLy)


def kAAtokskm_label(x, z=2.2):
    kstr = x
    knew = float(kstr) / (speed_light / (1 + z) / lambdaLy)
    transformed_label = "{:.3f}".format(knew)
    return transformed_label


def kskmtokAA_label(x, z=2.2):
    kstr = x
    knew = float(kstr) * (speed_light / (1 + z) / lambdaLy)
    transformed_label = "{:.3f}".format(knew)
    return transformed_label




def place_k_speed_unit_axis(fig, ax, fontsize=None, size=None, pos=0.2):
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
    # this createss more x-axes to compare things in k[s/km]
    par1 = ax.twiny()
    par1.set_xlabel(r" k [s/km] @ z=2.2", fontsize=fontt)
    par1.xaxis.set_major_formatter(FuncFormatter(partial(kskmtokAA_label, z)))


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
