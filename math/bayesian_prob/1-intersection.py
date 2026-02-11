#!/usr/bin/env python3
"""Intersection module"""

import numpy as np


def intersection(x, n, P, Pr):
    """Docstring."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if (not isinstance(Pr, np.ndarray)) or (Pr.shape != P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    log_nCk = (np.math.lgamma(n + 1) -
               np.math.lgamma(x + 1) -
               np.math.lgamma(n - x + 1))

    logP = np.empty_like(P, dtype=float)
    maskP = P > 0
    logP[maskP] = np.log(P[maskP])
    logP[~maskP] = -np.inf

    logQ = np.empty_like(P, dtype=float)
    maskQ = P < 1
    logQ[maskQ] = np.log(1 - P[maskQ])
    logQ[~maskQ] = -np.inf

    if x == 0:
        term_p = np.zeros_like(P, dtype=float)
    else:
        term_p = x * logP

    nx = n - x
    if nx == 0:
        term_q = np.zeros_like(P, dtype=float)
    else:
        term_q = nx * logQ

    logL = log_nCk + term_p + term_q
    L = np.exp(logL)

    return L * Pr
