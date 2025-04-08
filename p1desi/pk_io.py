import struct

import astropy.table as t
import fitsio
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.signal import savgol_filter
from scipy.stats import sem

try:
    import sgolay2
except ImportError:
    print("SGolay2 not installed, 2D smoothing of covariance matrix unavailable")

from p1desi import utils


def read_pk_means(pk_means_name, hdu=None):
    if hdu is None:
        table = t.Table.read(pk_means_name)
    else:
        table = t.Table.read(pk_means_name, hdu=hdu)
    return table


class Pk(object):
    def __init__(
        self,
        velunits=False,
        zbin=None,
        number_chunks=None,
        kmin=None,
        kmax=None,
        zmin=None,
        zmax=None,
        number_qso=None,
        mean_snr=None,
        mean_z=None,
        k=None,
        p=None,
        p_raw=None,
        p_noise=None,
        p_diff=None,
        err=None,
        err_raw=None,
        err_noise=None,
        err_diff=None,
        norm_p=None,
        norm_err=None,
        minrescor=None,
        maxrescor=None,
        p_noise_miss=None,
        resocor=None,
        err_resocor=None,
        cov_k1=None,
        cov_k2=None,
        cov=None,
        boot_cov=None,
    ):
        self.velunits = velunits
        self.zbin = zbin
        self.number_chunks = number_chunks
        self.kmin = kmin
        self.kmax = kmax
        self.zmin = zmin
        self.zmax = zmax
        self.number_qso = number_qso
        self.mean_snr = mean_snr
        self.mean_z = mean_z
        self.k = k
        self.p = p
        self.p_raw = p_raw
        self.p_noise = p_noise
        self.p_diff = p_diff
        self.err = err
        self.err_raw = err_raw
        self.err_noise = err_noise
        self.err_diff = err_diff
        self.norm_p = norm_p
        self.norm_err = norm_err
        self.minrescor = minrescor
        self.maxrescor = maxrescor
        self.p_noise_miss = p_noise_miss
        self.resocor = resocor
        self.err_resocor = err_resocor
        self.cov_k1 = cov_k1
        self.cov_k2 = cov_k2
        self.cov = cov
        self.boot_cov = boot_cov

    @classmethod
    def read_from_picca(cls, name_file,use_bootstrap_average_covariance=False):
        minrescor = {}
        maxrescor = {}
        number_chunks = {}
        kmin = {}
        kmax = {}
        zmin = {}
        zmax = {}
        mean_snr = {}
        mean_z = {}
        k = {}
        p = {}
        p_raw = {}
        p_noise = {}
        p_diff = {}
        err = {}
        err_raw = {}
        err_noise = {}
        err_diff = {}
        norm_p = {}
        norm_err = {}
        p_noise_miss = {}
        resocor = {}
        err_resocor = {}
        zbins = []
        cov_k1 = {}
        cov_k2 = {}
        cov = {}
        boot_cov = {}

        mean_pk = read_pk_means(name_file, hdu=1)
        metadata = read_pk_means(name_file, hdu=2)

        try:
            covariance = read_pk_means(name_file, hdu=3)
        except:
            covariance = None
        try:
            velunits = metadata.meta["VELUNITS"]
            number_qso = metadata.meta["NQSO"]
        except:
            velunits = False
            number_qso = 0

        minrescor_default = np.inf
        maxrescor_default = 0.0
        for i in range(len(metadata)):
            (w,) = np.where(mean_pk["index_zbin"] == i)
            zbin = mean_pk[w]["zbin"][0]
            zbins.append(zbin)
            k[zbin] = np.array(mean_pk[w]["meank"])
            p[zbin] = np.array(mean_pk[w]["meanPk"])
            p_raw[zbin] = np.array(mean_pk[w]["meanPk_raw"])
            p_noise[zbin] = np.array(mean_pk[w]["meanPk_noise"])
            p_diff[zbin] = np.array(mean_pk[w]["meanPk_diff"])
            err[zbin] = np.array(mean_pk[w]["errorPk"])
            err_raw[zbin] = np.array(mean_pk[w]["errorPk_raw"])
            err_noise[zbin] = np.array(mean_pk[w]["errorPk_noise"])
            err_diff[zbin] = np.array(mean_pk[w]["errorPk_diff"])
            norm_p[zbin] = np.array(mean_pk[w]["meanDelta2"])
            norm_err[zbin] = np.array(mean_pk[w]["errorDelta2"])
            number_chunks[zbin] = int(metadata[i]["N_chunks"])
            kmin[zbin] = metadata[i]["k_min"]
            kmax[zbin] = metadata[i]["k_max"]
            zmin[zbin] = metadata[i]["z_min"]
            zmax[zbin] = metadata[i]["z_max"]
            mean_snr[zbin] = np.mean(mean_pk[w]["meanforest_snr"])
            mean_z[zbin] = np.mean(mean_pk[w]["meanforest_z"])
            resocor[zbin] = np.array(mean_pk[w]["meancor_reso"])
            err_resocor[zbin] = np.array(mean_pk[w]["errorcor_reso"])

            if "meanPk_noise_miss" in mean_pk.colnames:
                p_noise_miss[zbin] = np.array(mean_pk[w]["meanPk_noise_miss"])
            else:
                p_noise_miss[zbin] = None

            if "rescor" in mean_pk.colnames:
                try:
                    if np.max(k[zbin]) > 0:
                        minrescor[zbin] = np.min(
                            [
                                minrescor_default,
                                np.min(
                                    k[zbin][
                                        (mean_pk["rescor"] < 0.1)
                                        & (mean_pk["rescor"] > 0)
                                    ]
                                ),
                            ]
                        )
                        maxrescor[zbin] = np.max(
                            [
                                maxrescor_default,
                                np.min(
                                    k[zbin][
                                        (mean_pk["rescor"] < 0.1)
                                        & (mean_pk["rescor"] > 0)
                                    ]
                                ),
                            ]
                        )
                except:
                    print("rescor information not computed, skipping")

                    minrescor[zbin] = minrescor_default
                    maxrescor[zbin] = maxrescor_default
            else:
                minrescor[zbin] = minrescor_default
                maxrescor[zbin] = maxrescor_default
            if covariance is not None:
                (w,) = np.where(covariance["index_zbin"] == i)
                cov_k1[zbin] = np.array(covariance[w]["k1"])
                cov_k2[zbin] = np.array(covariance[w]["k2"])
                cov[zbin] = np.array(covariance[w]["covariance"])
                if use_bootstrap_average_covariance:
                    boot_cov[zbin] = np.array(covariance[w]["boot_average_covariance"])
                else:
                    boot_cov[zbin] = np.array(covariance[w]["boot_covariance"])

        return cls(
            velunits=velunits,
            zbin=np.array(zbins),
            number_chunks=number_chunks,
            kmin=kmin,
            kmax=kmax,
            zmin=zmin,
            zmax=zmax,
            number_qso=number_qso,
            mean_snr=mean_snr,
            mean_z=mean_z,
            k=k,
            p=p,
            p_raw=p_raw,
            p_noise=p_noise,
            p_diff=p_diff,
            err=err,
            err_raw=err_raw,
            err_noise=err_noise,
            err_diff=err_diff,
            norm_p=norm_p,
            norm_err=norm_err,
            minrescor=minrescor,
            maxrescor=maxrescor,
            p_noise_miss=p_noise_miss,
            resocor=resocor,
            err_resocor=err_resocor,
            cov_k1=cov_k1,
            cov_k2=cov_k2,
            cov=cov,
            boot_cov=boot_cov,
        )

    @classmethod
    def read_from_p1desi(cls, name_file, zbins):
        minrescor = {}
        maxrescor = {}
        number_chunks = {}
        mean_snr = {}
        mean_z = {}
        k = {}
        p = {}
        p_raw = {}
        p_noise = {}
        p_diff = {}
        err = {}
        err_raw = {}
        err_noise = {}
        err_diff = {}
        norm_p = {}
        norm_err = {}
        p_noise_miss = {}
        resocor = {}
        err_resocor = {}

        mean_pk = read_pk_means(name_file)

        try:
            velunits = mean_pk.meta["VELUNITS"]
            number_qso = mean_pk.meta["NQSO"]
        except:
            velunits = False
            number_qso = 0

        minrescor_default = np.inf
        maxrescor_default = 0.0

        for iz, zbin in enumerate(zbins):
            dat = mean_pk[iz]
            select = dat["N"] > 0
            k[zbin] = np.array(dat["meank"][select])
            p[zbin] = np.array(dat["meanPk"][select])
            p_raw[zbin] = np.array(dat["meanPk_raw"][select])
            p_noise[zbin] = np.array(dat["meanPk_noise"][select])
            p_diff[zbin] = np.array(dat["meanPk_diff"][select])
            err[zbin] = np.array(dat["errorPk"][select])
            err_raw[zbin] = np.array(dat["errorPk_raw"][select])
            err_noise[zbin] = np.array(dat["errorPk_noise"][select])
            err_diff[zbin] = np.array(dat["errorPk_diff"][select])
            norm_p[zbin] = np.array(dat["meanDelta2"][select])
            norm_err[zbin] = np.array(dat["errorDelta2"][select])
            resocor[zbin] = np.array(dat["meancor_reso"][select])
            err_resocor[zbin] = np.array(dat["errorcor_reso"][select])
            number_chunks[zbin] = int(dat["N_chunks"])
            mean_snr[zbin] = dat["meansnr"]
            mean_z[zbin] = dat["meansnr"]

            if "meanPk_noise_miss" in dat.colnames:
                p_noise_miss[zbin] = np.array(dat["meanPk_noise_miss"][select])
            else:
                p_noise_miss[zbin] = None

            if "rescor" in dat.colnames:
                try:
                    if np.max(k[zbin]) > 0:
                        minrescor[zbin] = np.min(
                            [
                                minrescor_default,
                                np.min(
                                    k[zbin][
                                        (dat["rescor"][select] < 0.1)
                                        & (dat["rescor"][select] > 0)
                                    ]
                                ),
                            ]
                        )
                        maxrescor[zbin] = np.max(
                            [
                                maxrescor_default,
                                np.min(
                                    k[zbin][
                                        (dat["rescor"][select] < 0.1)
                                        & (dat["rescor"][select] > 0)
                                    ]
                                ),
                            ]
                        )
                except:
                    print("rescor information not computed, skipping")

                    minrescor[zbin] = minrescor_default
                    maxrescor[zbin] = maxrescor_default
            else:
                minrescor[zbin] = minrescor_default
                maxrescor[zbin] = maxrescor_default

        return cls(
            velunits=velunits,
            zbin=np.array(zbins),
            number_chunks=number_chunks,
            number_qso=number_qso,
            mean_snr=mean_snr,
            mean_z=mean_z,
            k=k,
            p=p,
            p_raw=p_raw,
            p_noise=p_noise,
            p_diff=p_diff,
            err=err,
            err_raw=err_raw,
            err_noise=err_noise,
            err_diff=err_diff,
            norm_p=norm_p,
            norm_err=norm_err,
            minrescor=minrescor,
            maxrescor=maxrescor,
            p_noise_miss=p_noise_miss,
            resocor=resocor,
            err_resocor=err_resocor,
        )

    def compute_additional_stats(self):
        err_noiseoverraw = {}
        err_diffoverraw = {}
        err_diffovernoise = {}

        for z in self.zbin:
            err_noiseoverraw[z] = (self.p_noise[z] / self.p_raw[z]) * np.sqrt(
                (self.err_noise[z] / self.p_noise[z]) ** 2
                + (self.err_raw[z] / self.p_raw[z]) ** 2
            )
            err_diffoverraw[z] = (self.p_diff[z] / self.p_raw[z]) * np.sqrt(
                (self.err_diff[z] / self.p_diff[z]) ** 2
                + (self.err_raw[z] / self.p_raw[z]) ** 2
            )
            err_diffovernoise[z] = (self.p_diff[z] / self.p_noise[z]) * np.sqrt(
                (self.err_noise[z] / self.p_noise[z]) ** 2
                + (self.err_diff[z] / self.p_diff[z]) ** 2
            )

        self.err_noiseoverraw = err_noiseoverraw
        self.err_diffoverraw = err_diffoverraw
        self.err_diffovernoise = err_diffovernoise

    def correct_noise_global(self, p_noise_miss_angstrom_global):
        for z in self.zbin:
            if self.velunits:
                p_noise_corr = utils.kskmtokAA(p_noise_miss_angstrom_global, z=z)
            else:
                p_noise_corr = p_noise_miss_angstrom_global
            self.p[z] = self.p[z] - (p_noise_corr / self.resocor[z])
            self.norm_p[z] = (
                self.norm_p[z] - self.k[z] * (p_noise_corr / self.resocor[z]) / np.pi
            )
            self.p_noise[z] = self.p_noise[z] - p_noise_corr

    def posify_covariance_diagonal(self):
        for _, z in enumerate(self.zbin):
            nkbins = len(self.k[z])
            covariance_matrix = self.cov[z].reshape(nkbins, nkbins)
            positive_diagonal_covariance = posify_variance(np.diag(covariance_matrix))
            np.fill_diagonal(covariance_matrix, positive_diagonal_covariance)
            self.cov[z] = np.ravel(covariance_matrix)
            if self.boot_cov is not None:
                boot_covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
                positive_diagonal_boot_covariance = posify_variance(
                    np.diag(boot_covariance_matrix)
                )
                np.fill_diagonal(
                    boot_covariance_matrix, positive_diagonal_boot_covariance
                )
                self.boot_cov[z] = np.ravel(boot_covariance_matrix)

    def regularize_covariance(
        self,
        eigval_min=1e-10,
    ):
        for _, z in enumerate(self.zbin):
            nkbins = len(self.k[z])
            covariance_matrix = self.cov[z].reshape(nkbins, nkbins)
            self.cov[z] = np.ravel(
                regularize_covariance(covariance_matrix, eigval_min=eigval_min)
            )
            if self.boot_cov is not None:
                boot_covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
                self.boot_cov[z] = np.ravel(
                    regularize_covariance(boot_covariance_matrix, eigval_min=eigval_min)
                )

    def smooth_covariance_diagonal(
        self,
        smoothing_window=50,
        smoothing_polynomial=5,
    ):
        for _, z in enumerate(self.zbin):
            nkbins = len(self.k[z])
            covariance_matrix = self.cov[z].reshape(nkbins, nkbins)
            smooth_covariance_diagonal = smooth_variance(
                np.diag(covariance_matrix),
                smoothing_window,
                smoothing_polynomial,
            )
            np.fill_diagonal(covariance_matrix, smooth_covariance_diagonal)
            self.cov[z] = np.ravel(covariance_matrix)
            if self.boot_cov is not None:
                boot_covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
                smooth_boot_covariance_diagonal = smooth_variance(
                    np.diag(boot_covariance_matrix),
                    smoothing_window,
                    smoothing_polynomial,
                )
                np.fill_diagonal(
                    boot_covariance_matrix, smooth_boot_covariance_diagonal
                )
                self.boot_cov[z] = np.ravel(boot_covariance_matrix)

    def smooth_covariance(
        self,
        smooth_covariance_window=15,
        smooth_covariance_order=5,
        remove_diagonal=True,
    ):

        for _, z in enumerate(self.zbin):
            nkbins = len(self.k[z])
            covariance_matrix = self.cov[z].reshape(nkbins, nkbins)
            self.cov[z] = np.ravel(
                smooth_covariance(
                    covariance_matrix,
                    smooth_covariance_window,
                    smooth_covariance_order,
                    remove_diagonal=remove_diagonal,
                )
            )
            if self.boot_cov is not None:
                boot_covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
                self.boot_cov[z] = np.ravel(
                    smooth_covariance(
                        boot_covariance_matrix,
                        smooth_covariance_window,
                        smooth_covariance_order,
                        remove_diagonal=remove_diagonal,
                    )
                )

    def multiplicative_covariance_correction(
        self,
        correction_term=0.9,
    ):
        for _, z in enumerate(self.zbin):
            if type(correction_term) == dict:
                correction = correction_term[z]
            else:
                correction = correction_term
            nkbins = len(self.k[z])
            covariance_matrix = self.cov[z].reshape(nkbins, nkbins)
            covariance_matrix *= correction
            self.cov[z] = np.ravel(covariance_matrix)
            if self.boot_cov is not None:
                boot_covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
                boot_covariance_matrix *= correction
                self.boot_cov[z] = np.ravel(boot_covariance_matrix)

    def correct_covariance(
        self,
        smoothing_variance_window=50,
        smoothing_variance_order=5,
        smooth_covariance_window=15,
        smooth_covariance_order=5,
        smooth_covariance_remove_diagonal=True,
        correction_covariance_term=0.9,
        eigval_min=1e-10,
    ):
        self.posify_covariance_diagonal()
        self.smooth_covariance_diagonal(
            smoothing_window=smoothing_variance_window,
            smoothing_polynomial=smoothing_variance_order,
        )
        self.smooth_covariance(
            smooth_covariance_window=smooth_covariance_window,
            smooth_covariance_order=smooth_covariance_order,
            remove_diagonal=smooth_covariance_remove_diagonal,
        )
        self.multiplicative_covariance_correction(
            correction_term=correction_covariance_term,
        )
        self.regularize_covariance(eigval_min=eigval_min)

    def use_covariance_as_error(
        self,
        use_boot=True,
    ):
        for _, z in enumerate(self.zbin):
            nkbins = len(self.k[z])
            if use_boot:
                covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
            else:
                covariance_matrix = self.cov[z].reshape(nkbins, nkbins)

            self.err[z] = np.sqrt(np.diag(covariance_matrix))
            self.norm_err[z] = self.k[z] * self.err[z] / np.pi


class PkEboss(Pk):
    def __init__(
        self,
        zbin=None,
        k=None,
        p=None,
        p_noise=None,
        err=None,
        norm_p=None,
        norm_err=None,
    ):
        super(PkEboss, self).__init__(
            velunits=True,
            zbin=zbin,
            number_chunks=None,
            k=k,
            p=p,
            p_noise=p_noise,
            err=err,
            norm_p=norm_p,
            norm_err=norm_err,
        )

    @classmethod
    def read_from_file(cls, name_file):
        k = {}
        p = {}
        p_noise = {}
        err = {}
        norm_p = {}
        norm_err = {}
        file_eboss = np.loadtxt(name_file)
        zbins = np.unique(file_eboss[:, 0])
        for zbin in zbins:
            mask = file_eboss[:, 0] == zbin
            k[zbin] = np.array(file_eboss[:, 1][mask])
            p[zbin] = np.array(file_eboss[:, 2][mask])
            norm_p[zbin] = np.array(k[zbin] * p[zbin] / np.pi)
            p_noise[zbin] = np.array(file_eboss[:, 4][mask])
            err_stat = file_eboss[:, 3][mask]
            err_syst = file_eboss[:, 6][mask]
            err_tot = np.sqrt(np.array(err_stat) ** 2 + np.array(err_syst) ** 2)
            err[zbin] = err_tot
            norm_err[zbin] = k[zbin] * err[zbin] / np.pi

        return cls(
            zbin=np.array(zbins),
            k=k,
            p=p,
            p_noise=p_noise,
            err=err,
            norm_p=norm_p,
            norm_err=norm_err,
        )


class PkHR(Pk):
    def __init__(
        self,
        zbin=None,
        k=None,
        p=None,
        err=None,
        norm_p=None,
        norm_err=None,
    ):
        super(PkHR, self).__init__(
            velunits=True,
            zbin=zbin,
            number_chunks=None,
            k=k,
            p=p,
            err=err,
            norm_p=norm_p,
            norm_err=norm_err,
        )

    @classmethod
    def read_from_file(cls, name_file):
        k = {}
        p = {}
        err = {}
        norm_p = {}
        norm_err = {}
        file_hr = np.loadtxt(name_file, delimiter="|", skiprows=1, usecols=(1, 2, 3, 4))
        zbins = np.unique(file_hr[:, 0])
        for zbin in zbins:
            mask = file_hr[:, 0] == zbin
            k[zbin] = np.array(file_hr[:, 1][mask])
            p[zbin] = np.array(file_hr[:, 2][mask])
            norm_p[zbin] = np.array(k[zbin] * p[zbin] / np.pi)
            err[zbin] = np.array(file_hr[:, 3][mask])
            norm_err[zbin] = np.array(k[zbin] * err[zbin] / np.pi)

        return cls(
            zbin=np.array(zbins),
            k=k,
            p=p,
            err=err,
            norm_p=norm_p,
            norm_err=norm_err,
        )


class PkTrueOhioMock(Pk):
    def __init__(
        self,
        zbin=None,
        k=None,
        p=None,
        err=None,
        norm_p=None,
        norm_err=None,
        velunits=None,
    ):
        super(PkTrueOhioMock, self).__init__(
            velunits=velunits,
            zbin=zbin,
            number_chunks=None,
            k=k,
            p=p,
            err=err,
            norm_p=norm_p,
            norm_err=norm_err,
        )

    @classmethod
    def read_from_file(cls, name_file, zbins_input, k_input, velunits=True):
        k = {}
        p = {}
        err = {}
        norm_p = {}
        norm_err = {}

        file = open(name_file, "rb")

        nk, nz = struct.unpack("ii", file.read(struct.calcsize("ii")))
        fmt = "d" * nz
        data = file.read(struct.calcsize(fmt))
        z_file = np.array(struct.unpack(fmt, data), dtype=np.double)
        fmt = "d" * nk
        data = file.read(struct.calcsize(fmt))
        k_file = np.array(struct.unpack(fmt, data), dtype=np.double)
        fmt = "d" * nk * nz
        data = file.read(struct.calcsize(fmt))
        p_file = np.array(struct.unpack(fmt, data), dtype=np.double).reshape((nz, nk))
        intp_p = RegularGridInterpolator(
            (k_file, z_file),
            np.transpose(p_file),
            method="linear",
            bounds_error=False,
        )
        if velunits:
            for zbin in zbins_input:
                k[zbin] = k_input[zbin]
                p[zbin] = intp_p((k_input[zbin], zbin))
                norm_p[zbin] = intp_p((k_input[zbin], zbin)) * k_input[zbin] / np.pi
                err[zbin] = np.zeros(k_input[zbin].shape)
                norm_err[zbin] = np.zeros(k_input[zbin].shape)
        else:
            for zbin in zbins_input:
                k_vel = utils.kAAtokskm(k_input[zbin], z=zbin)
                k[zbin] = k_input[zbin]
                p[zbin] = utils.kskmtokAA(intp_p((k_vel, zbin)), z=zbin)
                norm_p[zbin] = intp_p((k_vel, zbin)) * k_vel / np.pi
                err[zbin] = np.zeros(k_input[zbin].shape)
                norm_err[zbin] = np.zeros(k_input[zbin].shape)
        return cls(
            velunits=velunits,
            zbin=zbins_input,
            k=k,
            p=p,
            err=err,
            norm_p=norm_p,
            norm_err=norm_err,
        )


class ModelFitPk(object):
    def __init__(
        self,
        k=None,
        p=None,
        norm_p=None,
    ):
        self.k = k
        self.p = p
        self.norm_p = norm_p

    @classmethod
    def init_from_file(cls, filename, modelname, zmax=None):
        if modelname == "eBOSSmodel_stack":
            return cls.init_eboss_stack(filename)
        if modelname == "HRmodel_stack":
            return cls.init_hr_stack(filename)

    @classmethod
    def init_eboss_stack(cls, filename):
        def read_in_model(model_file):
            tab = fitsio.FITS(model_file)[1]
            z = tab["z"][:].reshape(-1, 1000)
            k = tab["k"][:].reshape(-1, 1000)
            kpk = tab["kpk"][:].reshape(-1, 1000)
            return z, k, kpk

        eBOSSmodel_lowz = read_in_model(filename[0])
        eBOSSmodel_highz = read_in_model(filename[1])
        eBOSSmodel_stack = [
            np.vstack([m, m2]) for m, m2 in zip(eBOSSmodel_lowz, eBOSSmodel_highz)
        ]
        z_unique = np.unique(eBOSSmodel_stack["z"])
        k, p, norm_p = {}, {}, {}
        for z in z_unique:
            mask = eBOSSmodel_stack["z"] == z
            k[z].append(eBOSSmodel_stack["k"][mask])
            norm_p[z].append(eBOSSmodel_stack["kpk"][mask])
            p[z].append(
                np.pi * eBOSSmodel_stack["kpk"][mask] / eBOSSmodel_stack["k"][mask]
            )

        return cls(k=k, p=p, norm_p=norm_p)

    @classmethod
    def init_hr_stack(cls, filename):
        def load_model(
            k,
            z,
            k0=0.009,
            k1=0.053,
            z0=3,
            A=0.066,
            B=3.59,
            n=-2.685,
            alpha=-0.22,
            beta=-0.16,
        ):
            knorm0 = k / k0
            knorm1 = k / k1
            exp1 = 3 + n + alpha * np.log(knorm0)
            exp2 = B + beta * np.log(knorm0)
            nom = knorm0**exp1
            denom = 1 + knorm1**2
            zfac = (1 + z) / (1 + z0)
            return A * nom / denom * zfac**exp2

        hrmodel = {}
        z_array = np.arange(2.2, 4.7, 0.2)
        k_array = np.arange(0.001, 0.1, 0.0001)
        hrmodel["kpk"] = load_model(
            k_array[np.newaxis, :],
            z_array[:, np.newaxis],
            A=0.084,
            B=3.64,
            alpha=-0.155,
            beta=0.32,
            k1=0.048,
            n=-2.655,
        )
        kk, zz = np.meshgrid(k_array, z_array)
        hrmodel["k"] = kk
        hrmodel["z"] = zz

        hrmodel_stack = (
            np.array(hrmodel["z"]),
            np.array(hrmodel["k"]),
            np.array(hrmodel["kpk"]),
        )

        z_unique = np.unique(hrmodel_stack["z"])
        k, p, norm_p = {}, {}, {}
        for z in z_unique:
            mask = hrmodel_stack["z"] == z
            k[z].append(hrmodel_stack["k"][mask])
            norm_p[z].append(hrmodel_stack["kpk"][mask])
            p[z].append(np.pi * hrmodel_stack["kpk"][mask] / hrmodel_stack["k"][mask])

        return cls(k=k, p=p, norm_p=norm_p)


class MeanPkZ(object):
    def __init__(
        self,
        velunits=False,
        k=None,
        p=None,
        p_raw=None,
        p_noise=None,
        p_diff=None,
        err=None,
        err_raw=None,
        err_noise=None,
        err_diff=None,
    ):
        self.velunits = velunits
        self.k = k
        self.p = p
        self.p_raw = p_raw
        self.p_noise = p_noise
        self.p_diff = p_diff
        self.err = err
        self.err_raw = err_raw
        self.err_noise = err_noise
        self.err_diff = err_diff

    @classmethod
    def init_from_pk(cls, pk, zmax=None):
        velunits = pk.velunits
        if zmax is None:
            zmax = np.inf

        nb_z_bins = len(pk.zbin[pk.zbin < zmax])

        k = np.mean([pk.k[z] for z in pk.zbin if z < zmax], axis=0)
        p = np.mean([pk.p[z] for z in pk.zbin if z < zmax], axis=0)
        p_raw = np.mean([pk.p_raw[z] for z in pk.zbin if z < zmax], axis=0)
        p_noise = np.mean([pk.p_noise[z] for z in pk.zbin if z < zmax], axis=0)
        p_diff = np.mean([pk.p_diff[z] for z in pk.zbin if z < zmax], axis=0)
        err = np.mean([pk.err[z] for z in pk.zbin if z < zmax], axis=0) / np.sqrt(
            nb_z_bins
        )
        err_raw = np.mean(
            [pk.err_raw[z] for z in pk.zbin if z < zmax], axis=0
        ) / np.sqrt(nb_z_bins)
        err_noise = np.mean(
            [pk.err_noise[z] for z in pk.zbin if z < zmax], axis=0
        ) / np.sqrt(nb_z_bins)
        err_diff = np.mean(
            [pk.err_diff[z] for z in pk.zbin if z < zmax], axis=0
        ) / np.sqrt(nb_z_bins)

        return cls(
            velunits, k, p, p_raw, p_noise, p_diff, err, err_raw, err_noise, err_diff
        )

    def compute_additional_stats(self, pk, zmax):
        nb_z_bins = len(pk.zbin[pk.zbin < zmax])

        err_noiseoverraw = np.mean(
            [
                (pk.p_noise[z] / pk.p_raw[z])
                * np.sqrt(
                    (pk.err_noise[z] / pk.p_noise[z]) ** 2
                    + (pk.err_raw[z] / pk.p_raw[z]) ** 2
                )
                for z in pk.zbin
                if z < zmax
            ],
            axis=0,
        ) / np.sqrt(nb_z_bins)
        err_diffoverraw = np.mean(
            [
                (pk.p_diff[z] / pk.p_raw[z])
                * np.sqrt(
                    (pk.err_diff[z] / pk.p_diff[z]) ** 2
                    + (pk.err_raw[z] / pk.p_raw[z]) ** 2
                )
                for z in pk.zbin
                if z < zmax
            ],
            axis=0,
        ) / np.sqrt(nb_z_bins)
        err_diffovernoise = np.mean(
            [
                (pk.p_diff[z] / pk.p_noise[z])
                * np.sqrt(
                    (pk.err_noise[z] / pk.p_noise[z]) ** 2
                    + (pk.err_diff[z] / pk.p_diff[z]) ** 2
                )
                for z in pk.zbin
                if z < zmax
            ],
            axis=0,
        ) / np.sqrt(nb_z_bins)

        self.err_noiseoverraw = err_noiseoverraw
        self.err_diffoverraw = err_diffoverraw
        self.err_diffovernoise = err_diffovernoise

    def compute_noise_asymptopte(self, k_asymptote, use_diff=False):
        mask_k = self.k > k_asymptote
        if use_diff:
            noise = self.p_diff
        else:
            noise = self.p_noise
        alpha = np.nanmean((self.p_raw - noise)[mask_k])
        beta = np.nanmean((noise / self.p_raw)[mask_k])
        return alpha, beta


class MeanPkK(object):
    def __init__(
        self,
        velunits=False,
        zbin=None,
        p=None,
        p_raw=None,
        p_noise=None,
        p_diff=None,
        err_noise=None,
        err_diff=None,
    ):
        self.velunits = velunits
        self.zbin = zbin
        self.p = p
        self.p_raw = p_raw
        self.p_noise = p_noise
        self.p_diff = p_diff
        self.err_noise = err_noise
        self.err_diff = err_diff

    @classmethod
    def init_from_pk(cls, pk, kmax=None):
        velunits = pk.velunits
        if kmax is None:
            kmax = np.inf

        p, p_raw, p_noise, p_diff, err_noise, err_diff = {}, {}, {}, {}, {}, {}
        for _, z in enumerate(pk.zbin):
            mask = pk.k[z] < kmax
            p[z] = np.mean(pk.p[z][mask])
            p_raw[z] = np.mean(pk.p_raw[z][mask])
            p_noise[z] = np.mean(pk.p_noise[z][mask])
            err_noise[z] = sem(pk.p_noise[z][mask], ddof=0)
            p_diff[z] = np.mean(pk.p_diff[z][mask])
            err_diff[z] = sem(pk.p_diff[z][mask], ddof=0)

        return cls(velunits, pk.zbin, p, p_raw, p_noise, p_diff, err_noise, err_diff)

    def compute_additional_stats(self, pk, kmax=None):
        noiseoverdiff, err_noiseoverdiff = {}, {}
        for _, z in enumerate(pk.zbin):
            mask = pk.k[z] < kmax
            noiseoverdiff[z] = np.mean(pk.p_noise[z][mask] / pk.p_diff[z][mask])

            err_noiseoverdiff[z] = (
                np.mean(pk.p_noise[z][mask]) / np.mean(pk.p_diff[z][mask])
            ) * np.sqrt(
                (self.err_noise[z] / np.mean(pk.p_noise[z][mask])) ** 2
                + (self.err_diff[z] / np.mean(pk.p_diff[z][mask])) ** 2
            )
        self.noiseoverdiff = noiseoverdiff
        self.err_noiseoverdiff = err_noiseoverdiff


def posify_variance(variance):
    if len(variance) == len(variance[np.isnan(variance)]):
        return variance
    elif len(variance) == len(variance[variance < 0.0]):
        return variance
    else:
        mask_negative_variance = variance < 0.0
        variance_indices = np.arange(len(variance))
        interp_func = interp1d(
            variance_indices[~mask_negative_variance],
            variance[~mask_negative_variance],
            kind="linear",
            fill_value="extrapolate",
        )
        variance_positive = interp_func(variance_indices)
        return variance_positive


def smooth_variance(
    variance,
    smoothing_window,
    smoothing_polynomial,
):

    window_filter = min(
        smoothing_window,
        int(3 * len(variance) / 4),
    )
    variance_smooth = np.exp(
        savgol_filter(
            np.log(variance),
            window_filter,
            smoothing_polynomial,
        )
    )
    return variance_smooth


def smooth_covariance(
    covariance,
    smooth_cov_window,
    smooth_cov_order,
    remove_diagonal=True,
):
    if remove_diagonal:
        diag = np.copy(np.diag(covariance))
        diag_for_smoothing = np.concatenate(
            [
                [(covariance[0, 1] + covariance[1, 0]) / 2],
                [
                    (covariance[i - 1, i + 1] + covariance[i + 1, i - 1]) / 2
                    for i in range(1, len(diag) - 1)
                ],
                [(covariance[-1, -2] + covariance[-2, -1]) / 2],
            ]
        )
        np.fill_diagonal(covariance, diag_for_smoothing)

    covariance_smooth = sgolay2.SGolayFilter2(
        window_size=smooth_cov_window,
        poly_order=smooth_cov_order,
    )(covariance)

    if remove_diagonal:
        np.fill_diagonal(covariance_smooth, diag)
    return covariance_smooth


def regularize_covariance(
    covariance,
    eigval_min=1e-10,
):
    if np.isfinite(covariance).all():
        eigval, eigvec = np.linalg.eig(covariance)
        eigval[eigval < 0.0] = eigval_min
        covariance_regularized = eigvec @ np.diag(eigval) @ np.linalg.inv(eigvec)
        return covariance_regularized
    return covariance
