#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def P_init(X, perplexity):
    """A function that does the trick"""
    n, d = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    D = (np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X))
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
