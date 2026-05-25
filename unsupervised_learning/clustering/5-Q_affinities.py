#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def Q_affinities(Y):
    """A function that does the trick"""
    n = Y.shape[0]
    sum_Y = np.sum(np.square(Y), axis=1)
    D = -2 * np.dot(Y, Y.T)
    D = D + sum_Y.reshape(-1, 1) + sum_Y.reshape(1, -1)

    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    denominator = np.sum(num)
    Q = num / denominator
    return Q, num
