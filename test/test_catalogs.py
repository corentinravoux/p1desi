"""Unit tests for :mod:`p1desi.catalogs` (QSO/DLA catalog FITS I/O)."""

import numpy as np
import pandas as pd
import pytest

from p1desi import catalogs

# ``return_dataframe`` calls ``ndarray.newbyteorder()``, which NumPy 2.0 removed.
# The reader therefore raises under NumPy >= 2; pin that as a known limitation
# rather than editing the (frozen) package code.
NUMPY2 = int(np.__version__.split(".")[0]) >= 2
numpy2_reader_xfail = pytest.mark.xfail(
    NUMPY2,
    reason="catalogs.return_dataframe uses ndarray.newbyteorder(), removed in NumPy 2.0",
    raises=AttributeError,
    strict=True,
)


def _make_df(n=5, targetid_start=0):
    return pd.DataFrame(
        {
            "TARGETID": np.arange(targetid_start, targetid_start + n),
            "RA": np.linspace(10.0, 20.0, n),
            "DEC": np.linspace(-5.0, 5.0, n),
            "Z": np.linspace(2.1, 3.5, n),
        }
    )


@numpy2_reader_xfail
def test_save_and_read_roundtrip(workdir):
    df = _make_df()
    catalogs.save_dataframe_to_fits(df, "cat.fits", extname="QSO_CAT")
    back = catalogs.return_dataframe("cat.fits", extname="QSO_CAT")
    assert list(back.columns) == list(df.columns)
    np.testing.assert_array_equal(back["TARGETID"].to_numpy(), df["TARGETID"].to_numpy())
    np.testing.assert_allclose(back["RA"].to_numpy(), df["RA"].to_numpy())


@numpy2_reader_xfail
def test_return_dataframe_drop_columns(workdir):
    catalogs.save_dataframe_to_fits(_make_df(), "cat.fits")
    back = catalogs.return_dataframe("cat.fits", colname_drop=["DEC"])
    assert "DEC" not in back.columns
    assert "RA" in back.columns


@numpy2_reader_xfail
def test_return_dataframe_unique(workdir):
    df = pd.concat([_make_df(3), _make_df(3)], ignore_index=True)  # duplicated ids
    catalogs.save_dataframe_to_fits(df, "cat.fits")
    back = catalogs.return_dataframe("cat.fits", return_unique=True)
    assert len(back) == 3
    assert len(np.unique(back["TARGETID"])) == 3


def test_merge_dataframes():
    merged = catalogs.merge_panda_dataframe([_make_df(2), _make_df(3, targetid_start=100)])
    assert len(merged) == 5


def test_get_number(workdir, capsys):
    catalogs.save_dataframe_to_fits(_make_df(7), "cat.fits", extname="QSO_CAT")
    catalogs.get_number("cat.fits", type="QSO")
    out = capsys.readouterr().out
    assert "7" in out
