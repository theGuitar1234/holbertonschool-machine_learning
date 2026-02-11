#!/usr/bin/env python3
"""Docstring for 0-likelihood.py"""

import numpy as np


def likelihood(x, n, P):
    """Docstring for likelihood."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    log_nCk = (np.math.lgamma(n + 1) -
               np.math.lgamma(x + 1) -
               np.math.lgamma(n - x + 1))

    with np.errstate(divide='ignore'):
        logL = log_nCk + x * np.log(P) + (n - x) * np.log(1 - P)

    L = np.exp(logL)

    L[(P == 0) & (x > 0)] = 0
    L[(P == 1) & (n - x > 0)] = 0

    return L
