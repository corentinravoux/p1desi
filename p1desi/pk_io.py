"""Reading, representing and post-processing the P1D measurement.

Central I/O module of :mod:`p1desi`. Defines the :class:`Pk` container that
holds a mean one-dimensional Lyman-alpha power spectrum and all of its
auxiliary arrays (raw / noise / diff power, errors, resolution correction,
covariance) as dictionaries keyed by redshift bin, plus reader classmethods for
the picca and p1desi FITS formats and for a handful of external measurements
(eBOSS, high-resolution, Ohio mocks). Also provides covariance
regularisation/smoothing helpers and per-redshift / per-k reductions.
"""

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

from p1desi import corrections, utils


def read_pk_means(pk_means_name, hdu=None):
    """Read a P1D table from a FITS file as an astropy Table.

    Args:
        pk_means_name (str): FITS file path.
        hdu (int, optional): HDU index to read (default HDU if None).

    Returns:
        astropy.table.Table: The requested table.
    """
    if hdu is None:
        table = t.Table.read(pk_means_name)
    else:
        table = t.Table.read(pk_means_name, hdu=hdu)
    return table


class Pk(object):
    """A mean one-dimensional Lyman-alpha power spectrum and its diagnostics.

    All spectral quantities are stored as dictionaries keyed by redshift bin
    (``self.zbin`` is the array of bin centres). For a bin ``z``, ``k[z]`` are
    the wavenumbers (Angstrom^-1 unless ``velunits`` is True, then s/km),
    ``p[z]`` the corrected power, ``p_raw``/``p_noise``/``p_diff`` the raw /
    noise / exposure-difference power, ``err*`` the matching errors,
    ``norm_p``/``norm_err`` the dimensionless ``k*P/pi`` power, ``resocor`` /
    ``err_resocor`` the resolution correction, and ``cov`` / ``boot_cov`` the
    analytic / bootstrap covariance (indexed by ``cov_k1``/``cov_k2``).

    Instances are normally built through the ``read_from_*`` classmethods rather
    than the constructor.
    """

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
        """Store the P1D arrays and metadata (all keyed by redshift bin).

        See the class docstring for the meaning of each quantity. Every
        argument defaults to None so partially populated instances are allowed.
        """
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
    def read_from_picca(cls, name_file, use_bootstrap_average_covariance=False):
        """Build a :class:`Pk` from a picca mean-P1D FITS file.

        Reads the mean-P1D table (HDU 1), the per-bin metadata (HDU 2) and, when
        present, the covariance (HDU 3), filling every per-redshift dictionary.

        Args:
            name_file (str): picca mean-P1D FITS file.
            use_bootstrap_average_covariance (bool, optional): Use the
                ``boot_average_covariance`` column instead of ``boot_covariance``.

        Returns:
            Pk: The populated power-spectrum object.
        """
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
        """Build a :class:`Pk` from a p1desi-format mean-P1D FITS file.

        Unlike :meth:`read_from_picca`, the p1desi format stores one table row
        per redshift bin (indexed by ``zbins``) with per-k arrays and an ``N``
        occupancy column used to drop empty k bins.

        Args:
            name_file (str): p1desi mean-P1D FITS file.
            zbins (list[float]): Redshift bin centres, in table row order.

        Returns:
            Pk: The populated power-spectrum object (no covariance).
        """
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
        """Compute error-propagated ratios of the noise/diff/raw power spectra.

        Populates ``self.err_noiseoverraw``, ``self.err_diffoverraw`` and
        ``self.err_diffovernoise`` (per redshift bin) by propagating the
        individual errors through each ratio.
        """
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
        """Subtract a global missing-noise power from the P1D in place.

        The missing-noise term (given in Angstrom^-1 units) is converted to the
        object's units, divided by the resolution correction and removed from
        ``p``, ``norm_p`` and ``p_noise`` for every redshift bin.

        Args:
            p_noise_miss_angstrom_global (float): Global missing-noise power in
                Angstrom^-1 units.
        """
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
        """Force the covariance diagonal positive for every redshift bin.

        Applies :func:`posify_variance` to the diagonal of ``cov`` (and
        ``boot_cov`` if present), in place.
        """
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
        """Clip the covariance eigenvalues from below for every redshift bin.

        Applies :func:`regularize_covariance` to ``cov`` (and ``boot_cov`` if
        present), in place, to keep the matrix positive definite.

        Args:
            eigval_min (float, optional): Minimum allowed eigenvalue.
        """
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
        """Savitzky-Golay-smooth the covariance diagonal per redshift bin.

        Applies :func:`smooth_variance` to the diagonal of ``cov`` (and
        ``boot_cov`` if present), in place.

        Args:
            smoothing_window (int, optional): Savitzky-Golay window length.
            smoothing_polynomial (int, optional): Savitzky-Golay polynomial order.
        """
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
        """2D-smooth the off-diagonal covariance per redshift bin.

        Applies :func:`smooth_covariance` to ``cov`` (and ``boot_cov`` if
        present), in place; requires ``sgolay2`` for the 2D filtering.

        Args:
            smooth_covariance_window (int, optional): 2D smoothing window.
            smooth_covariance_order (int, optional): 2D polynomial order.
            remove_diagonal (bool, optional): Preserve the diagonal, smoothing
                only the off-diagonal structure.
        """
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
        """Rescale the covariance by a multiplicative factor, in place.

        Args:
            correction_term (float | dict, optional): Scalar factor, or a per-
                redshift dictionary of factors.
        """
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
        """Run the full covariance cleaning pipeline in place.

        Chains, per redshift bin: diagonal posification, diagonal smoothing,
        off-diagonal 2D smoothing, multiplicative rescaling and eigenvalue
        regularisation.

        Args:
            smoothing_variance_window (int, optional): Diagonal smoothing window.
            smoothing_variance_order (int, optional): Diagonal smoothing order.
            smooth_covariance_window (int, optional): Off-diagonal 2D window.
            smooth_covariance_order (int, optional): Off-diagonal 2D order.
            smooth_covariance_remove_diagonal (bool, optional): Keep the diagonal
                untouched during 2D smoothing.
            correction_covariance_term (float | dict, optional): Multiplicative
                rescaling factor.
            eigval_min (float, optional): Minimum eigenvalue for regularisation.
        """
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
        """Replace the diagonal errors by the covariance diagonal, in place.

        Sets ``err`` to the square root of the (bootstrap or analytic)
        covariance diagonal and updates the normalised ``norm_err``.

        Args:
            use_boot (bool, optional): Use the bootstrap covariance (else the
                analytic one).
        """
        for _, z in enumerate(self.zbin):
            nkbins = len(self.k[z])
            if use_boot:
                covariance_matrix = self.boot_cov[z].reshape(nkbins, nkbins)
            else:
                covariance_matrix = self.cov[z].reshape(nkbins, nkbins)

            self.err[z] = np.sqrt(np.diag(covariance_matrix))
            self.norm_err[z] = self.k[z] * self.err[z] / np.pi


class PkEboss(Pk):
    """A :class:`Pk` loaded from the published eBOSS P1D text file (velocity units)."""

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
        """Store the eBOSS P1D arrays (see :class:`Pk`); ``velunits`` is True."""
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
        """Load the eBOSS P1D from its multi-column text table.

        The statistical and systematic error columns are combined in quadrature.

        Args:
            name_file (str): eBOSS P1D text file.

        Returns:
            PkEboss: The loaded power spectrum.
        """
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
    """A :class:`Pk` loaded from a high-resolution P1D text file (velocity units)."""

    def __init__(
        self,
        zbin=None,
        k=None,
        p=None,
        err=None,
        norm_p=None,
        norm_err=None,
    ):
        """Store the high-resolution P1D arrays (see :class:`Pk`)."""
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
        """Load the high-resolution P1D from its pipe-delimited text table.

        Args:
            name_file (str): High-resolution P1D text file.

        Returns:
            PkHR: The loaded power spectrum.
        """
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
    """The true (input) P1D of the Ohio mocks, read from a binary grid file."""

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
        """Store the mock true-P1D arrays (see :class:`Pk`)."""
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
        """Load the Ohio-mock true P1D and interpolate it onto a target grid.

        Reads the ``(nk, nz)`` binary grid, builds a linear ``(k, z)``
        interpolator and evaluates it at the requested bins/wavenumbers,
        converting units when ``velunits`` is False. Errors are set to zero.

        Args:
            name_file (str): Binary mock-P1D file.
            zbins_input (array-like): Target redshift bins.
            k_input (dict): Target wavenumbers per redshift bin.
            velunits (bool, optional): Output in velocity units (s/km).

        Returns:
            PkTrueOhioMock: The interpolated true power spectrum.
        """
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
    """A tabulated P1D best-fit model (eBOSS or high-resolution stack)."""

    def __init__(
        self,
        k=None,
        p=None,
        norm_p=None,
    ):
        """Store the model wavenumbers ``k``, power ``p`` and normalised ``norm_p``."""
        self.k = k
        self.p = p
        self.norm_p = norm_p

    @classmethod
    def init_from_file(cls, filename, modelname, zmax=None):
        """Dispatch to the eBOSS or high-resolution model loader by name.

        Args:
            filename: File(s) holding the model.
            modelname (str): ``"eBOSSmodel_stack"`` or ``"HRmodel_stack"``.
            zmax (float, optional): Maximum redshift (unused by the loaders).

        Returns:
            ModelFitPk: The loaded model.
        """
        if modelname == "eBOSSmodel_stack":
            return cls.init_eboss_stack(filename)
        if modelname == "HRmodel_stack":
            return cls.init_hr_stack(filename)

    @classmethod
    def init_eboss_stack(cls, filename):
        """Load and stack the low- and high-redshift eBOSS best-fit models.

        Args:
            filename (sequence[str]): ``(low_z_file, high_z_file)`` FITS models.

        Returns:
            ModelFitPk: The stacked model keyed by redshift.
        """
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
        """Build the high-resolution analytic P1D model on a fixed (k, z) grid.

        Args:
            filename: Unused (the model is analytic, evaluated on a built-in grid).

        Returns:
            ModelFitPk: The model keyed by redshift.
        """
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
    """The P1D averaged over redshift bins (a single k-vector power spectrum)."""

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
        """Store the redshift-averaged P1D arrays (single vectors, not z-keyed)."""
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
        """Average a :class:`Pk` over its redshift bins below ``zmax``.

        Errors are averaged and reduced by ``sqrt(n_zbins)``.

        Args:
            pk (Pk): Source power spectrum.
            zmax (float, optional): Upper redshift bound (exclusive).

        Returns:
            MeanPkZ: The redshift-averaged power spectrum.
        """
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
        """Compute redshift-averaged error-propagated noise/diff/raw ratios.

        Populates ``self.err_noiseoverraw``, ``self.err_diffoverraw`` and
        ``self.err_diffovernoise`` from the source :class:`Pk`.

        Args:
            pk (Pk): Source power spectrum.
            zmax (float): Upper redshift bound (exclusive).
        """
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
        """Estimate the high-k noise asymptote of the redshift-averaged P1D.

        Args:
            k_asymptote (float): Wavenumber above which the asymptote is measured.
            use_diff (bool, optional): Use the exposure-difference power instead
                of the pipeline noise power.

        Returns:
            tuple[float, float]: ``(alpha, beta)`` — the mean raw-minus-noise
            power and the mean noise-over-raw ratio above ``k_asymptote``.
        """
        mask_k = self.k > k_asymptote
        if use_diff:
            noise = self.p_diff
        else:
            noise = self.p_noise
        alpha = np.nanmean((self.p_raw - noise)[mask_k])
        beta = np.nanmean((noise / self.p_raw)[mask_k])
        return alpha, beta


class MeanPkK(object):
    """The P1D averaged over wavenumbers (one value per redshift bin)."""

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
        """Store the k-averaged P1D scalars, keyed by redshift bin."""
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
        """Average a :class:`Pk` over wavenumbers below ``kmax`` per redshift bin.

        Args:
            pk (Pk): Source power spectrum.
            kmax (float, optional): Upper wavenumber bound (exclusive).

        Returns:
            MeanPkK: The k-averaged power spectrum (one scalar per redshift bin).
        """
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
        """Compute the k-averaged noise-over-diff ratio and its error per z bin.

        Populates ``self.noiseoverdiff`` and ``self.err_noiseoverdiff``.

        Args:
            pk (Pk): Source power spectrum.
            kmax (float, optional): Upper wavenumber bound (exclusive).
        """
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
    """Replace negative variance entries by a linear interpolation of the rest.

    All-NaN or all-negative inputs are returned unchanged.

    Args:
        variance (numpy.ndarray): Variance values (may contain negatives/NaNs).

    Returns:
        numpy.ndarray: Variance with negative entries interpolated away.
    """
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
    """Savitzky-Golay-smooth a variance vector in log space.

    NaN entries are preserved; the window is clamped to the data length and
    forced odd, and smoothing is skipped when too few valid points remain.

    Args:
        variance (numpy.ndarray): Variance values (positive; may contain NaNs).
        smoothing_window (int): Requested Savitzky-Golay window length.
        smoothing_polynomial (int): Savitzky-Golay polynomial order.

    Returns:
        numpy.ndarray: The smoothed variance.
    """
    window_filter = min(
        smoothing_window,
        int(3 * len(variance) / 4),
    )
    mask = np.isnan(variance)
    variance_smooth = np.zeros_like(variance)
    variance_smooth[mask] = np.nan
    valid_count = len(variance) - np.sum(mask)

    if window_filter % 2 == 0:
        window_filter -= 1

    if (
        valid_count >= window_filter
        and window_filter > smoothing_polynomial
        and window_filter >= 3
    ):
        variance_smooth[~mask] = np.exp(
            savgol_filter(
                np.log(variance[~mask]),
                window_filter,
                smoothing_polynomial,
            )
        )
    else:
        variance_smooth[~mask] = variance[~mask]

    return variance_smooth


def smooth_covariance(
    covariance,
    smooth_cov_window,
    smooth_cov_order,
    remove_diagonal=True,
):
    """2D Savitzky-Golay-smooth a covariance matrix (requires ``sgolay2``).

    When ``remove_diagonal`` is True the diagonal is temporarily replaced by a
    near-diagonal average during smoothing and restored afterwards, so only the
    off-diagonal structure is smoothed.

    Args:
        covariance (numpy.ndarray): Square covariance matrix.
        smooth_cov_window (int): 2D filter window size.
        smooth_cov_order (int): 2D polynomial order.
        remove_diagonal (bool, optional): Preserve the original diagonal.

    Returns:
        numpy.ndarray: The smoothed covariance matrix.
    """
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
    """Clip a covariance matrix's negative eigenvalues to a small positive floor.

    Non-finite matrices are returned unchanged.

    Args:
        covariance (numpy.ndarray): Square covariance matrix.
        eigval_min (float, optional): Floor applied to negative eigenvalues.

    Returns:
        numpy.ndarray: The regularised (positive-definite) covariance matrix.
    """
    if np.isfinite(covariance).all():
        eigval, eigvec = np.linalg.eig(covariance)
        eigval[eigval < 0.0] = eigval_min
        covariance_regularized = eigvec @ np.diag(eigval) @ np.linalg.inv(eigvec)
        return covariance_regularized
    return covariance


def print_infos(
    mean_pk,
    z,
    k_range_small,
    k_range_med,
    k_range_high,
    ref=None,
):
    """Print and return P1D summary values in three wavenumber ranges.

    Reports the mean power and error in a small-, medium- and high-k window at a
    single redshift, optionally as ratios to a reference dictionary.

    Args:
        mean_pk (Pk): Power spectrum to summarise.
        z (float): Redshift bin.
        k_range_small (tuple[float, float]): Small-k ``(min, max)`` window.
        k_range_med (tuple[float, float]): Medium-k window.
        k_range_high (tuple[float, float]): High-k window.
        ref (dict, optional): Reference values to divide by (keys ``pk_small``,
            ``pk_med``, ``pk_high``, ``err_small``, ``err_med``, ``err_high``).

    Returns:
        dict: The computed summary values (and their ratios to ``ref`` if given).
    """
    number_chunks = mean_pk.number_chunks[z]

    mask_k_small = (mean_pk.k[z] >= k_range_small[0]) & (
        mean_pk.k[z] < k_range_small[1]
    )
    mask_k_med = (mean_pk.k[z] >= k_range_med[0]) & (mean_pk.k[z] < k_range_med[1])
    mask_k_high = (mean_pk.k[z] >= k_range_high[0]) & (mean_pk.k[z] < k_range_high[1])

    pk_small = mean_pk.p[z][mask_k_small]
    pk_med = mean_pk.p[z][mask_k_med]
    pk_high = mean_pk.p[z][mask_k_high]
    err_small = mean_pk.err[z][mask_k_small]
    err_med = mean_pk.err[z][mask_k_med]
    err_high = mean_pk.err[z][mask_k_high]

    print("number of chunks = {}".format(number_chunks))
    print("pk at small k = {}".format(np.mean(pk_small)))
    print("pk at medium k = {}".format(np.mean(pk_med)))
    print("pk at high k = {}".format(np.mean(pk_high)))
    print("err at small k = {}".format(np.mean(err_small)))
    print("err at medium k = {}".format(np.mean(err_med)))
    print("err at high k = {}".format(np.mean(err_high)))

    if ref is not None:
        print(
            "Ratio to reference at small k = {}".format(
                np.mean(pk_small) / ref["pk_small"]
            )
        )
        print(
            "Ratio to reference at medium k = {}".format(
                np.mean(pk_med) / ref["pk_med"]
            )
        )
        print(
            "Ratio to reference at high k = {}".format(
                np.mean(pk_high) / ref["pk_high"]
            )
        )
        print(
            "Ratio to reference err at small k = {}".format(
                np.mean(err_small) / ref["err_small"]
            )
        )
        print(
            "Ratio to reference err at medium k = {}".format(
                np.mean(err_med) / ref["err_med"]
            )
        )
        print(
            "Ratio to reference err at high k = {}".format(
                np.mean(err_high) / ref["err_high"]
            )
        )
        return {
            "number_chunks": number_chunks,
            "pk_small": np.mean(pk_small),
            "pk_med": np.mean(pk_med),
            "pk_high": np.mean(pk_high),
            "err_small": np.mean(err_small),
            "err_med": np.mean(err_med),
            "err_high": np.mean(err_high),
            "ratio_pk_small": np.mean(pk_small) / ref["pk_small"],
            "ratio_pk_med": np.mean(pk_med) / ref["pk_med"],
            "ratio_pk_high": np.mean(pk_high) / ref["pk_high"],
            "ratio_err_small": np.mean(err_small) / ref["err_small"],
            "ratio_err_med": np.mean(err_med) / ref["err_med"],
            "ratio_err_high": np.mean(err_high) / ref["err_high"],
        }

    return {
        "number_chunks": number_chunks,
        "pk_small": np.mean(pk_small),
        "pk_med": np.mean(pk_med),
        "pk_high": np.mean(pk_high),
        "err_small": np.mean(err_small),
        "err_med": np.mean(err_med),
        "err_high": np.mean(err_high),
    }


def load_and_apply_correction(
    file_pk,
    p_noise_miss,
    zmax,
    correction_to_apply,
    file_correction_hcd,
    file_correction_bal,
    file_correction_lines,
    file_correction_cont,
    file_correction_resolution,
    file_metal,
    use_bootstrap_average_covariance=False,
):
    """Read a picca P1D, clean its covariance and apply all corrections.

    Convenience end-to-end entry point: reads the measurement, optionally
    subtracts a global missing-noise term, cleans the covariance and adopts it
    as the error, then applies the DESI mask/continuum and (if a metal file is
    given) side-band corrections via :func:`corrections.apply_p1d_corections`.

    Args:
        file_pk (str): picca mean-P1D FITS file.
        p_noise_miss (float | None): Global missing-noise power to subtract.
        zmax (float): Maximum redshift processed.
        correction_to_apply (list[str]): Correction names to apply.
        file_correction_hcd (str): HCD correction file.
        file_correction_bal (str): BAL correction file.
        file_correction_lines (str): Sky-line correction file.
        file_correction_cont (str): Continuum correction file.
        file_correction_resolution (str): Resolution correction file.
        file_metal (str | None): Side-band / metal correction file (None disables
            the side-band correction).
        use_bootstrap_average_covariance (bool, optional): Passed through to
            :meth:`Pk.read_from_picca`.

    Returns:
        Pk: The corrected power spectrum.
    """
    mean_pk = Pk.read_from_picca(
        file_pk, use_bootstrap_average_covariance=use_bootstrap_average_covariance
    )
    if p_noise_miss is not None:
        mean_pk.correct_noise_global(p_noise_miss)
    mean_pk.correct_covariance()
    mean_pk.use_covariance_as_error()

    apply_DESI_maskcont_corr = True
    apply_eBOSS_maskcont_corr = False
    if file_metal is None:
        apply_DESI_sb_corr = False
    else:
        apply_DESI_sb_corr = True
    apply_eBOSS_sb_corr = False

    corrections.apply_p1d_corections(
        mean_pk,
        zmax,
        apply_DESI_maskcont_corr,
        apply_eBOSS_maskcont_corr,
        apply_DESI_sb_corr,
        apply_eBOSS_sb_corr,
        correction_to_apply,
        file_correction_hcd=file_correction_hcd,
        file_correction_bal=file_correction_bal,
        file_correction_lines=file_correction_lines,
        file_correction_cont=file_correction_cont,
        file_correction_resolution=file_correction_resolution,
        file_metal=file_metal,
    )

    return mean_pk
