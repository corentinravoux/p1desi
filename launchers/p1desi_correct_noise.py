#!/usr/bin/env python3
import fitsio, os, glob
import numpy as np
from p1desi import bookkeeping, corrections

qsocat = "/global/cfs/cdirs/desi/users/ravouxco/pk1d/desi_data/fugu/p1d_healpix/v6/qsocat/catalog_fugu_afterburner_nqt_nobal_BI.fits"

path_in = (
    "/global/cfs/cdirs/desi/users/ravouxco/pk1d/desi_data/fugu/p1d_healpix/v6/p1d_out/"
)

lines_name = "DESIfuji2.5"
catalog_name = "fugu_afterburner_nqt_nobal_BI"
dla_name = "allcombine20.3"
bal = "None"
noise_estimate = "pipeline"
suffix = "_v6"
suffix_out = "_v6_noise_correction"


snr_cuts = 1

regions = ["lya", "SB1", "SB2"]

noise_corrections = [
    {  # "sv1": (0.026,1.77,0.00076),
        # "sv2": (0.0,0.0,0.00127),
        "sv3": (0.0, 0.0, 0.00127),
        # "special": (0.0,0.0,0.00127),
        "main": (0.0, 0.0, 0.00109),
    },
    {  # "sv1": [0.018,1.52,0.000032],
        # "sv2": [0.0,0.0,0.00048],
        "sv3": [0.0, 0.0, 0.00048],
        # "special": [0.0,0.0,0.00048],
        "main": [0.0, 0.0, 0.00019],
    },
    {  # "sv1": [0.018,1.52,0.000032],
        # "sv2": [0.0,0.0,0.00048],
        "sv3": [0.0, 0.0, 0.00048],
        # "special": [0.0,0.0,0.00048],
        "main": [0.0, 0.0, 0.00019],
    },
]


if __name__ == "__main__":
    for i in range(len(regions)):
        region = regions[i]
        noise_correction = noise_corrections[i]

        for snr_cut in snr_cuts:

            folder_name = bookkeeping.return_folder_name(
                region, lines_name, catalog_name, dla_name, bal, noise_estimate, suffix
            )
            pk_in = bookkeeping.return_pk_path(path_in, folder_name, snr_cut)
            folder_name_out = bookkeeping.return_folder_name(
                region,
                lines_name,
                catalog_name,
                dla_name,
                bal,
                noise_estimate,
                suffix_out,
            )
            pk_out = bookkeeping.return_pk_path(path_in, folder_name_out, snr_cut)
            corrections.correct_individual_pk_noise(
                pk_in, pk_out, qsocat, noise_correction
            )
