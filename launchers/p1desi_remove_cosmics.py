#!/usr/bin/env python3
import fitsio,os, shutil
snr_cuts = [1,2,3]
path = "/global/cfs/cdirs/desi/users/ravouxco/pk1d/desi_data/fugu/p1d_healpix/v6/p1d_out/p1d_lya_noisepipeline_linesDESIfuji2.5_catfugu_afterburner_nqt_nobal_BI_dlaallcombine20.3_balNone_v6_noise_correction"
targetid_to_remove = 39627631729051509

for snr_cut in snr_cuts:
    file_to_change = os.path.join(path,f"pk1d_SNRcut{snr_cut}/Pk1D-93.fits.gz")
    file_out = os.path.join(path,f"pk1d_SNRcut{snr_cut}/Pk1D-93_corr.fits.gz")

    f  = fitsio.FITS(file_to_change)
    out = fitsio.FITS(file_out,"rw",clobber=True)
    for i in range(1,len(f)):
        if f[i].read_header()["LOS_ID"] == targetid_to_remove:
            print(f[i].read_header()["LOS_ID"],i)
            continue
        else:
            pk = f[i].read()
            head = f[i].read_header()
            out.write(pk,header=head)
    out.close()

    os.makedirs(os.path.join(path,f"pk1d_SNRcut{snr_cut}/pk_cosmic_issue"),exist_ok=True)
    shutil.move(file_to_change,os.path.join(path,f"pk1d_SNRcut{snr_cut}/pk_cosmic_issue"))