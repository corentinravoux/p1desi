import fitsio
import pandas as pd
import numpy as np




def return_dataframe(catalog_name,extname='QSO_CAT',colname_drop=None,return_unique=False):
    if (colname_drop is not None):
        f = fitsio.FITS(catalog_name)[extname].read().byteswap().newbyteorder()
        dtype_new = []
        for i in range(len(f.dtype.names)):
            if(f.dtype.names[i] not in colname_drop):
                dtype_new.append((f.dtype.names[i],f.dtype[f.dtype.names[i]]))
        new_f = np.zeros(f.shape[0], dtype=dtype_new)
        for i in range(len(dtype_new)):
            new_f[dtype_new[i][0]] = f[dtype_new[i][0]]
        df = pd.DataFrame(new_f)
    else:
        df = pd.DataFrame(fitsio.FITS(catalog_name)[extname].read().byteswap().newbyteorder())
    if(return_unique):
        _, idx = np.unique(df["TARGETID"],return_index=True)
        sel = np.zeros(df['TARGETID'].size, dtype='bool')
        sel[idx] = True
        return(df[sel])
    else:
        return(df)


def merge_panda_dataframe(list_data):
    merged_dataframe = pd.concat(list_data,ignore_index=True)
    return merged_dataframe


def save_dataframe_to_fits(dataframe, filename, extname="QSO_CAT", clobber=True):
    """
    Save info from pandas dataframe in a fits file.
    Args:
        dataframe (pandas dataframe): dataframe containg the all the necessary QSO info
        filename (str):  name of the fits file
        clobber (bool):  overwrite the fits file defined by filename ?
    Returns:
        None
    """
    fits = fitsio.FITS(filename, 'rw', clobber=clobber)
    fits.write(dataframe.to_records(index=False), extname=extname)
    fits.close()
