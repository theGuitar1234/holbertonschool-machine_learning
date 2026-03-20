#!/usr/bin/env python3
'''
batch normalization
'''

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
    normalizes an unactivated output of a neural network using batch
    normalization

    Parameters:
    Z: numpy.ndarray of shape (m, n) containing the unactivated output of
    the previous layer
    gamma: numpy.ndarray of shape (1, n) containing the scales used for
    batch normalization
    beta: numpy.ndarray of shape (1, n) containing the offsets used for
    batch normalization
    epsilon: small number to avoid division by zero

    Returns:
    The normalized Z matrix
    '''
    # Calculate the mean and variance of Z
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    # Scale and shift Z_norm using gamma and beta
    Z_scaled_shifted = gamma * Z_norm + beta

    return Z_scaled_shifted
