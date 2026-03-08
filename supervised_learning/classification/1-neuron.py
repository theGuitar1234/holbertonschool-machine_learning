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
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
