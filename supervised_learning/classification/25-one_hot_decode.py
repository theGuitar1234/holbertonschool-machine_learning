#!/usr/bin/env python3
"""
defines function that converts a one-hot matrix
into a vector of labels
"""


import numpy as np


def one_hot_decode(one_hot):
    """
    one_hot_decode
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    vector = one_hot.transpose().argmax(axis=1)
    return vector
