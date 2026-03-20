#!/usr/bin/env python3
'''
shuffle the data points
'''

import numpy as np


def shuffle_data(X, Y):
    """
    shuffling_data
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffle = X[permutation]
    Y_shuffle = Y[permutation]
    return X_shuffle, Y_shuffle
