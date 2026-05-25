#!/usr/bin/env python3
"""GMM maximization step"""
import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None

    probs = np.sum(g, axis=0)
    if not np.isclose(np.sum(probs), n):
        return None, None, None

    Nk = np.sum(g, axis=1)
    pi = Nk / n
    m = np.matmul(g, X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        x_m = X - m[i]
        S[i] = np.matmul(g[i] * x_m.T, x_m) / Nk[i]

    return pi, m, S
