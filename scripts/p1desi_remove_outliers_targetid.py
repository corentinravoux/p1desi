#!/usr/bin/env python3
import glob
import multiprocessing as mp
import os
import shutil
from functools import partial

import fitsio
import numpy as np
from p1desi import bookkeeping

global_path = bookkeeping.get_path("global_path")
(
    noise_estimate,
    catalog_name,
    lines_name,
    dla_name,
    bal,
) = bookkeeping.get_params("param")
path = os.path.join(global_path, "data/p1d_out")
suffix = f"_noise_correction"
snr_cut_path = 1

regions = ["SB2"]

targetid_to_remove = [
    39627631729051509,
    39628511102634286,
    39628315232833588,
    39628512784550454,
    39633338260000189,
    39636538107103557,
    39627433195864992,
]

number_process = 1


def contains_targetid(target_id, pk_name):
    pk = fitsio.FITS(pk_name)
    targetid = np.array([int(pk[j].read_header()["LOS_ID"]) for j in range(1, len(pk))])
    mask = np.in1d(targetid, target_id)
    if len(mask[mask]) > 0:
        return True
    return False


for region in regions:
    folder_name = bookkeeping.return_folder_name(
        region, lines_name, catalog_name, dla_name, bal, noise_estimate, suffix
    )

    ### Find pk files to treat

    pk_path = os.path.join(path, f"p1d_{folder_name}", f"pk1d_SNRcut{snr_cut_path}")
    pk_files = glob.glob(os.path.join(pk_path, "Pk*"))

    if number_process == 1:
        pk_to_modify = []
        for i in range(len(pk_files)):
            pk = fitsio.FITS(pk_files[i])
            targetid = np.array(
                [int(pk[j].read_header()["LOS_ID"]) for j in range(1, len(pk))]
            )
            mask = np.in1d(targetid, targetid_to_remove)
            if len(mask[mask]) > 0:
                pk_to_modify.append(pk_files[i])
    else:
        func = partial(contains_targetid, targetid_to_remove)
        with mp.Pool(number_process) as pool:
            boolean_mask = pool.map(func, pk_files)
        pk_to_modify = pk_files[np.array(boolean_mask)]

    ### Modify pk

    for i in range(len(pk_to_modify)):
        file_out = pk_to_modify[i].split(".fits.gz")[0] + "_corr.fits.gz"
        if (os.path.isfile(pk_to_modify[i]) is False) & (os.path.isfile(file_out)):
            print("Already treated")
        else:
            f = fitsio.FITS(pk_to_modify[i])
            out = fitsio.FITS(file_out, "rw", clobber=True)
            for j in range(1, len(f)):
                if f[j].read_header()["LOS_ID"] in targetid_to_remove:
                    print(
                        f"Removing {f[j].read_header()['LOS_ID']} in {pk_to_modify[i]}"
                    )
                    continue
                else:
                    pk = f[j].read()
                    head = f[j].read_header()
                    out.write(pk, header=head)
            out.close()

            os.makedirs(os.path.join(pk_path, "pk_outliers"), exist_ok=True)
            shutil.move(
                pk_to_modify[i],
                os.path.join(pk_path, "pk_outliers"),
            )
