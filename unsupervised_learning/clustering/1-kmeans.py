#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    K-means on a data set
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer, max number of iterations
    Returns: C, clss, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    C = np.random.uniform(X_min, X_max, size=(k, d))

    clss = None
    for i in range(iterations):
        centroids = np.copy(C)
        distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(X_min, X_max, size=(d,))
            else:
                C[c] = X[clss == c].mean(axis=0)

        if (centroids == C).all():
            break

    # final assignment with updated centroids
    distances = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    clss = np.argmin(distances, axis=0)

    return C, clss
