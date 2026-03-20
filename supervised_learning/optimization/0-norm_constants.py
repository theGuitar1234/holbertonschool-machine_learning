#!/usr/bin/env python3
'''
normalizing constants
'''

import numpy as np


def normalization_constants(X):
    """
    normalization constants
    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
