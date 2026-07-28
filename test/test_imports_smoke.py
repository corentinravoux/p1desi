"""Smoke tests: every module byte-compiles and imports (where deps allow).

``skylines`` and ``spectra_statistics`` import ``desispec`` / ``speclite`` at
module load; those are only needed to read raw DESI spectra and may be absent
(or, as with some ``scipy``/``speclite`` combinations, broken), so their import
is skipped gracefully while their syntax is still checked.
"""

import importlib
import py_compile

import pytest

from conftest import PACKAGE_ROOT

CORE_MODULES = [
    "utils", "hcd", "catalogs", "bookkeeping", "mocks", "resolution",
    "pk_io", "plotpk", "corrections", "uncertainty", "metals", "noise",
]
OPTIONAL_MODULES = ["skylines", "spectra_statistics"]

ALL_PY = sorted((PACKAGE_ROOT / "p1desi").rglob("*.py")) \
    + sorted((PACKAGE_ROOT / "scripts").glob("*.py")) \
    + sorted((PACKAGE_ROOT / "launchers").glob("*.py"))


@pytest.mark.parametrize("path", ALL_PY, ids=[p.name for p in ALL_PY])
def test_module_compiles(path):
    """Every shipped .py file byte-compiles (no syntax errors)."""
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
    """Core modules import cleanly."""
    importlib.import_module(f"p1desi.{module}")


@pytest.mark.parametrize("module", OPTIONAL_MODULES)
def test_optional_module_imports(module):
    """Optional (raw-spectrum) modules import when their heavy deps are present."""
    try:
        importlib.import_module(f"p1desi.{module}")
    except Exception as exc:  # desispec / speclite missing or broken
        pytest.skip(f"optional dependency unavailable: {exc!r}")
