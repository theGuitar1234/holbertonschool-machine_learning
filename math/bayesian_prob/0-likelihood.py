#!/usr/bin/env python3
"""Likelihood module"""

import numpy as np


def likelihood(x, n, P):
    """Docstring for likelihood."""

    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError("x must be a positive integer")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    log_nCk = (np.math.lgamma(n + 1) -
               np.math.lgamma(x + 1) -
               np.math.lgamma(n - x + 1))

    if x == 0:
        term_p = np.zeros_like(P, dtype=float)
    else:
        term_p = x * np.where(P == 0, -np.inf, np.log(P))

    nx = n - x
    if nx == 0:
        term_q = np.zeros_like(P, dtype=float)
    else:
        term_q = nx * np.where(P == 1, -np.inf, np.log(1 - P))

    logL = log_nCk + term_p + term_q
    return np.exp(logL)
