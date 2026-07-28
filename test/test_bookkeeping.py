"""Unit tests for :mod:`p1desi.bookkeeping` (path / name conventions)."""

import os

from p1desi import bookkeeping


def test_return_folder_name_encodes_variant():
    name = bookkeeping.return_folder_name(
        "lya", "DESIfuji2.5", "mycat", "mydla", "None", "pipeline", suffix="_v6"
    )
    assert name == (
        "lya_noisepipeline_linesDESIfuji2.5_catmycat_dlamydla_balNone_v6"
    )


def test_return_folder_name_none_suffix():
    a = bookkeeping.return_folder_name("lya", "l", "c", "d", "0", "pipeline", "None")
    b = bookkeeping.return_folder_name("lya", "l", "c", "d", "0", "pipeline", None)
    assert a == b
    assert a.endswith("bal0")  # no trailing suffix


def test_return_delta_and_pk_paths():
    folder = "lya_noisepipeline_linesL_catC_dlaD_balNone_v6"
    delta = bookkeeping.return_delta_path("/base", folder)
    pk = bookkeeping.return_pk_path("/base", folder, 1)
    assert delta == os.path.join("/base", f"p1d_{folder}", "deltas")
    assert pk == os.path.join("/base", f"p1d_{folder}", "pk1d_SNRcut1")


def test_return_mean_pk_name_variants():
    assert bookkeeping.return_mean_pk_name(False, False) == "mean_Pk1d.fits.gz"
    assert bookkeeping.return_mean_pk_name(True, False) == "mean_Pk1d_vel.fits.gz"
    assert bookkeeping.return_mean_pk_name(False, True) == "mean_Pk1d_log.fits.gz"
    assert (
        bookkeeping.return_mean_pk_name(True, True, snr_cut_mean=2)
        == "mean_Pk1d_log_vel_snr_cut_mean.fits.gz"
    )


def test_get_path_and_params(tmp_path):
    p = tmp_path / "global_path"
    p.write_text("/data/root\n")
    assert bookkeeping.get_path(str(p)) == "/data/root"

    param = tmp_path / "param"
    param.write_text("pipeline\nmycat\nmylines\nmydla\nNone\n")
    noise, cat, lines, dla, bal = bookkeeping.get_params(str(param))
    assert (noise, cat, lines, dla, bal) == (
        "pipeline", "mycat", "mylines", "mydla", "None"
    )
