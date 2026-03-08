#!/usr/bin/env python3
"""script to define a single neurion performing binary classification"""

import numpy as np


class Neuron:
    """class Neuron that defines a single neuron performing binary class."""
    def __init__(self, nx):
        """
        binary classification
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        returns weights
        """
        return self.__W

    @property
    def b(self):
        """
        returns bias
        """
        return self.__b

    @property
    def A(self):
        """
        returns output
        """
        return self.__A

    def forward_prop(self, X):
        """
        returns propagation of neuron
        """
        E = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-E))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        returns cost of logistic regression
        """
        m = Y.shape[1]
        C = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return C
