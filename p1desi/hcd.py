"""High column density (HCD / DLA) contamination model for the P1D.

Implements the Rogers et al. redshift- and scale-dependent correction ratio
used to model the residual effect of high column density absorbers on the
one-dimensional Lyman-alpha power spectrum.
"""

import numpy as np


def rogers(z, k, a0, a1, b0, b1, c0, c1):
    """Rogers et al. HCD contamination ratio as a function of redshift and scale.

    Each of the ``a``, ``b``, ``c`` parameters evolves with redshift as a power
    law pivoted at ``z0 = 2.0``; the returned ratio multiplies the P1D to model
    the residual high column density contribution.

    Args:
        z (float | array-like): Redshift.
        k (float | array-like): Wavenumber (Angstrom^-1).
        a0, a1 (float): Amplitude term value at ``z0`` and its redshift index.
        b0, b1 (float): Exponential-scale term value at ``z0`` and its index.
        c0, c1 (float): Additive term value at ``z0`` and its redshift index.

    Returns:
        float | array-like: The HCD contamination ratio at ``(z, k)``.
    """
    z0 = 2.0
    a = a0 * ((1 + z) / (1 + z0)) ** a1
    b = b0 * ((1 + z) / (1 + z0)) ** b1
    c = c0 * ((1 + z) / (1 + z0)) ** c1
    d = 1 * ((1 + z) / (1 + z0)) ** (-3.55)

    ratio = d * (1 / (a * np.exp(b * k) - 1) ** 2) + c
    return ratio
