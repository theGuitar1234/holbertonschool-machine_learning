#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def pca(X, var=0.95):
    """A function that does the trick"""
    u, Sigma, vh = np.linalg.svd(X, full_matrices=False)
    cumulative_var = np.cumsum(Sigma) / np.sum(Sigma)
    r = (np.argwhere(cumulative_var >= var))[0, 0]
    w = vh.T
    wr = w[:, :r + 1]
    return wr
