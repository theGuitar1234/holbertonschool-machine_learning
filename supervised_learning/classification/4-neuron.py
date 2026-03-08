#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """Single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for W"""
        return self.__W

    @property
    def b(self):
        """Getter for b"""
        return self.__b

    @property
    def A(self):
        """Getter for A"""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression"""
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions"""
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
