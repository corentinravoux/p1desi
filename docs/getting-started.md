# Getting started

## Installation

```bash
git clone https://github.com/corentinravoux/p1desi.git
cd p1desi
pip install .            # or: pip install -e ".[dev]" for tests
```

The core dependencies (`fitsio`, `numpy`, `scipy`, `matplotlib`) are installed
automatically. A few modules pull in optional extras:

- `pandas` — [`catalogs`](api/catalogs.md);
- `astropy` — used by [`pk_io`](api/pk_io.md) for FITS table I/O;
- `sgolay2` — 2-D covariance smoothing in [`pk_io`](api/pk_io.md) /
  [`uncertainty`](api/uncertainty.md) (a warning is printed if missing);
- `desispec` / `speclite` — only [`spectra_statistics`](api/spectra_statistics.md)
  and [`skylines`](api/skylines.md), which read raw DESI spectra.

## Loading a P1D

The central object is [`pk_io.Pk`](api/pk_io.md). Read a picca mean-P1D file:

```python
from p1desi import pk_io

pk = pk_io.Pk.read_from_picca("mean_Pk1d_vel.fits.gz")

print(pk.zbin)            # redshift bin centres
print(pk.velunits)        # True if k is in s/km, False for Angstrom^-1

z = pk.zbin[0]
k   = pk.k[z]             # wavenumbers
p   = pk.p[z]             # corrected power
err = pk.err[z]           # error
```

Every spectral quantity is a dictionary keyed by redshift bin: `p_raw`,
`p_noise`, `p_diff`, `resocor`, `norm_p` (the dimensionless `k·P/π`), and — when
the file has a covariance HDU — `cov` / `boot_cov` (indexed by `cov_k1`,
`cov_k2`).

## Applying the full correction chain

```python
from p1desi import pk_io

pk = pk_io.load_and_apply_correction(
    file_pk="mean_Pk1d_vel.fits.gz",
    p_noise_miss=None,
    zmax=4.2,
    correction_to_apply=["resolution", "lines", "cont"],
    file_correction_hcd="hcd.fits",
    file_correction_bal="bal.fits",
    file_correction_lines="lines.fits",
    file_correction_cont="cont.fits",
    file_correction_resolution="reso.fits",
    file_metal="metal_sb.fits",
)
```

This reads the measurement, cleans and adopts the covariance as the error, and
applies the DESI mask/continuum plus side-band corrections. See the
[analysis pipeline](pipeline.md) for what each stage does.

## Plotting

```python
from p1desi import plotpk

plotpk.plot({"DESI": pk}, zmax=4.2, outname="p1d", outpoints="p1d_points")
```

## Bookkeeping

Analysis products are laid out by variant (region, line list, catalog, DLA/BAL
masking, noise estimate). [`bookkeeping`](api/bookkeeping.md) builds the paths:

```python
from p1desi import bookkeeping

folder = bookkeeping.return_folder_name(
    "lya", "DESIfuji2.5", "mycat", "mydla", "None", "pipeline", suffix="_v6"
)
pk_dir = bookkeeping.return_pk_path("/data/p1d_out", folder, snr_cut_path=1)
```
