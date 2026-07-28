"""QSO / DLA catalog helpers built on FITS and pandas.

Load DESI quasar (or DLA) catalogs from FITS into pandas dataframes, merge
several of them, write a dataframe back to FITS, and report object counts.
"""

import fitsio
import numpy as np
import pandas as pd


def return_dataframe(
    catalog_name,
    extname="QSO_CAT",
    colname_drop=None,
    return_unique=False,
):
    """Load a FITS catalog HDU into a pandas dataframe.

    Byte-swaps the FITS records to native order, optionally drops columns and
    optionally keeps only the first row of each unique ``TARGETID``.

    Args:
        catalog_name (str): Catalog FITS file path.
        extname (str, optional): HDU name to read. Defaults to ``"QSO_CAT"``.
        colname_drop (list[str], optional): Column names to exclude.
        return_unique (bool, optional): Keep only one row per ``TARGETID``.

    Returns:
        pandas.DataFrame: The (optionally filtered) catalog.
    """
    if colname_drop is not None:
        f = fitsio.FITS(catalog_name)[extname].read().byteswap().newbyteorder()
        dtype_new = []
        for i in range(len(f.dtype.names)):
            if f.dtype.names[i] not in colname_drop:
                dtype_new.append((f.dtype.names[i], f.dtype[f.dtype.names[i]]))
        new_f = np.zeros(f.shape[0], dtype=dtype_new)
        for i in range(len(dtype_new)):
            new_f[dtype_new[i][0]] = f[dtype_new[i][0]]
        df = pd.DataFrame(new_f)
    else:
        df = pd.DataFrame(
            fitsio.FITS(catalog_name)[extname].read().byteswap().newbyteorder()
        )
    if return_unique:
        _, idx = np.unique(df["TARGETID"], return_index=True)
        sel = np.zeros(df["TARGETID"].size, dtype="bool")
        sel[idx] = True
        return df[sel]
    else:
        return df


def merge_panda_dataframe(
    list_data,
):
    """Concatenate several catalog dataframes into one.

    Args:
        list_data (list[pandas.DataFrame]): Dataframes to concatenate.

    Returns:
        pandas.DataFrame: The row-wise concatenation with a fresh index.
    """
    merged_dataframe = pd.concat(list_data, ignore_index=True)
    return merged_dataframe


def save_dataframe_to_fits(
    dataframe,
    filename,
    extname="QSO_CAT",
    clobber=True,
    units=None,
    add_checksum=False,
):
    """
    Save info from pandas dataframe in a fits file.
    Args:
        dataframe (pandas dataframe): dataframe containg the all the necessary QSO info
        filename (str):  name of the fits file
        clobber (bool):  overwrite the fits file defined by filename ?
    Returns:
        None
    """
    fits = fitsio.FITS(filename, "rw", clobber=clobber)
    if units is not None:
        fits.write(
            dataframe.to_records(index=False),
            extname=extname,
            units=units,
        )
    else:
        fits.write(
            dataframe.to_records(index=False),
            extname=extname,
        )
    if add_checksum:
        fits[-1].write_checksum()
        fits[-1].verify_checksum()
    fits.close()


def get_number(
    catalog,
    type="QSO",
    extname=None,
):
    """Print the number of rows in a catalog HDU.

    Args:
        catalog (str): Catalog FITS file path.
        type (str, optional): ``"QSO"`` or ``"DLA"``, selecting the default HDU
            name when ``extname`` is not given. Defaults to ``"QSO"``.
        extname (str, optional): Explicit HDU name (overrides ``type``).
    """
    if type == "QSO":
        if extname is None:
            extname = "QSO_CAT"
    if type == "DLA":
        if extname is None:
            extname = "DLACAT"
    try:
        print(fitsio.FITS(catalog)[extname].get_nrows())
    except:
        print("Probably extname issue, catalog return", fitsio.FITS(catalog))
