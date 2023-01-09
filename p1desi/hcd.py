import numpy as np


def rogers(z, k, a0, a1, b0, b1, c0, c1):
    z0 = 2.0
    a = a0 * ((1 + z) / (1 + z0)) ** a1
    b = b0 * ((1 + z) / (1 + z0)) ** b1
    c = c0 * ((1 + z) / (1 + z0)) ** c1
    d = 1 * ((1 + z) / (1 + z0)) ** (-3.55)

    ratio = d * (1 / (a * np.exp(b * k) - 1) ** 2) + c
    return ratio
