#!/usr/bin/env python3
"""
defines function that converts a numeric label vector
into a one-hot matrix
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    one_hot_encode
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y].transpose()
        return one_hot
    except Exception as err:
        return None
