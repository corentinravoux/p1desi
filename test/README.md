# p1desi test suite

Run from the package root:

```bash
pip install -e ".[dev]"   # installs pytest alongside the runtime deps
pytest -q                 # or: pytest test/
```

p1desi ships no reference P1D FITS files, so the suite **synthesises** a
format-faithful picca mean-P1D file (three HDUs: mean table, per-bin metadata,
covariance) via `make_picca_pk_fits` in [`conftest.py`](conftest.py) and reads
it back through the public `Pk` API. Every file-writing test runs in a
temporary directory.

| File | Scope | What it covers |
|---|---|---|
| `test_utils.py` | unit | k-unit conversions (Å⁻¹ ↔ s/km) round-trips, `return_key`, tick-label formatters, doublet constants |
| `test_hcd.py` | unit | Rogers HCD contamination ratio, including the z = z₀ closed form |
| `test_resolution.py` | unit | resolution × pixelisation model and the `dl` fit (incl. NaN handling) |
| `test_bookkeeping.py` | unit | folder / delta / P1D path builders and mean-P1D file names |
| `test_catalogs.py` | unit | FITS catalog write / merge / count round-trips |
| `test_mocks.py` | unit | χ² of a measurement vs itself (zero) and vs an offset, per-redshift χ², `getChi2Pdf` |
| `test_pk_io.py` | unit | variance/covariance helpers, `Pk` covariance methods, `MeanPkZ`/`MeanPkK`, eBOSS/HR readers |
| `test_pk_io_e2e.py` | end-to-end | synthetic picca FITS → `Pk.read_from_picca` → reductions + covariance-cleaning pipeline |
| `test_imports_smoke.py` | smoke | every shipped module byte-compiles; core modules import; optional raw-spectrum modules import when their deps are present |

## Known environment limitations (pinned, not worked around)

- **`catalogs.return_dataframe`** calls `ndarray.newbyteorder()`, which **NumPy
  2.0 removed**. The three reader round-trip tests are marked `xfail` (strict)
  under NumPy ≥ 2, so they flip to `xpass` if the code is fixed or NumPy is
  pinned below 2.
- **`skylines` / `spectra_statistics`** import `desispec` / `speclite`, which
  can be missing or (with recent SciPy) broken; their import tests `skip`
  gracefully while their syntax is still checked.
- The 2-D covariance-cleaning test **skips** when `sgolay2` is not installed.

## Note on the solver-less parts

The full covariance-cleaning pipeline (`Pk.correct_covariance`) needs `sgolay2`
for 2-D smoothing; the individual steps that do not (posify / diagonal smoothing
/ rescale / regularise) are exercised separately so the path is covered either
way.
