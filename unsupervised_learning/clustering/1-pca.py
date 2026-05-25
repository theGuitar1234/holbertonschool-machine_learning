#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def pca(X, ndim):
    """A function that does the trick"""
    X_mean = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X_mean)
    W = V.T
    Wr = W[:, :ndim]
    T = np.dot(X_mean, Wr)
    return T
