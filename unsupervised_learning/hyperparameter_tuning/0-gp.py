#!/usr/bin/env python3
"""
hyperparameter tuning
"""

import numpy as np


class GaussianProcess:
    '''
    Gaussian Process class for regression.
    '''
    def __init__(self, X_init, y_init, l=1, sigma_f=1):
        '''
        Initialize the Gaussian Process with initial data and hyperparameters.

        Parameters:
        X_init (np.ndarray): Initial input data of shape
                            (n_samples, n_features).
        y_init (np.ndarray): Initial output data of shape
                            (n_samples,).
        l (float): Length scale hyperparameter.
        sigma_f (float): Output variance hyperparameter.
        '''
        self.l = l
        self.sigma_f = sigma_f
        self.X = X_init
        self.Y = y_init
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        '''
        Compute the kernel matrix between two sets of input data.

        Parameters:
        X1 (np.ndarray): First input data of shape
                            (n_samples, n_features).
        X2 (np.ndarray): Second input data of shape
                            (n_samples, n_features).

        Returns:
        np.ndarray: Kernel matrix of shape (n_samples, n_samples).
        '''
        n1, d = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = np.exp(
                    -0.5 * np.sum((X1[i] - X2[j]) ** 2) / self.l ** 2
                )
        return K * self.sigma_f ** 2
