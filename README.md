# p1desi

Suite of tools to analyse the **one-dimensional Lyman-alpha forest power
spectrum (P1D)** measured by [picca](https://github.com/igmhub/picca) on DESI
data. Developed for the first DESI P1D measurements (Early Data Release and
beyond).

📖 **Full documentation (API reference + user guide):** **<https://corentinravoux.github.io/p1desi/>**

---

## What it does

Starting from picca's per-chunk P1D outputs, p1desi reads and represents a mean
P1D, applies the instrumental and astrophysical corrections, builds the error
budget, validates the result and plots it.

```
DESI spectra ─▶ picca (deltas + per-chunk P1D)
             ─▶ p1desi.pk_io        mean P1D + covariance
             ─▶ corrections         noise / resolution / lines / metals / HCD
             ─▶ uncertainty         statistical + systematic covariance
             ─▶ plotpk / mocks      plots + validation
```

## Install

```bash
pip install .              # core deps: fitsio, numpy, scipy, matplotlib
pip install -e ".[dev]"    # + pytest, to run the test suite
```

Some modules use optional extras: `pandas` (catalogs), `astropy` (FITS tables in
`pk_io`), `sgolay2` (2-D covariance smoothing), and `desispec`/`speclite` (only
the modules that read raw DESI spectra).

## Quick start

```python
from p1desi import pk_io, plotpk

# read a picca mean-P1D FITS file
pk = pk_io.Pk.read_from_picca("mean_Pk1d_vel.fits.gz")

z = pk.zbin[0]
k, p, err = pk.k[z], pk.p[z], pk.err[z]   # dicts keyed by redshift bin

plotpk.plot({"DESI": pk}, zmax=4.2, outname="p1d", outpoints="p1d_points")
```

Apply the whole correction chain in one call:

```python
pk = pk_io.load_and_apply_correction(
    "mean_Pk1d_vel.fits.gz", p_noise_miss=None, zmax=4.2,
    correction_to_apply=["resolution", "lines", "cont"],
    file_correction_hcd="hcd.fits", file_correction_bal="bal.fits",
    file_correction_lines="lines.fits", file_correction_cont="cont.fits",
    file_correction_resolution="reso.fits", file_metal="metal_sb.fits",
)
```

## Package map

| Module | Role |
|---|---|
| `pk_io` | The `Pk` container (mean P1D, raw/noise/diff power, covariance) and its readers; covariance cleaning |
| `corrections` | Assemble and apply the correction chain |
| `uncertainty` | Statistical + systematic error budget and covariance plots |
| `metals` | Side-band metal (e.g. Si III) contamination fit and subtraction |
| `noise` | Noise-power ratios and high-k asymptote diagnostics |
| `resolution` | Spectrograph resolution × pixelisation correction |
| `skylines` | Sky-line identification and masking |
| `hcd` | Rogers et al. high-column-density (HCD/DLA) correction |
| `spectra_statistics` | Statistics of the input spectra / deltas |
| `mocks` | χ² / KS validation against mocks and reference measurements |
| `plotpk` | Plots of the P1D, ratios, residuals and comparisons |
| `bookkeeping` | Folder / file naming for analysis variants |
| `catalogs` | QSO / DLA catalog I/O helpers |
| `utils` | Constants and wavenumber-unit conversions (Å⁻¹ ↔ s/km) |

`scripts/` holds small CLIs (bookkeeping, catalog counts, outlier removal) and
`launchers/` holds edit-then-run analysis drivers.

## Conventions

- **Units.** Wavenumbers are in **Angstrom⁻¹** or **velocity units (s/km)**; the
  `Pk.velunits` flag records which. `utils.kAAtokskm` / `utils.kskmtokAA`
  convert. The dimensionless power `Δ² = k·P/π` is stored as `norm_p`.
- **Per-redshift dictionaries.** All spectral arrays in `Pk` are dictionaries
  keyed by redshift bin (`pk.zbin`).

## Documentation

📖 **Online: <https://corentinravoux.github.io/p1desi/>**

The site is built from the in-code docstrings with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) +
[mkdocstrings](https://mkdocstrings.github.io/) and deployed to GitHub Pages on
every push to `main`. Build it locally with:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

See [`test/README.md`](test/README.md) for the layout of the suite.
