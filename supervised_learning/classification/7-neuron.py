#!/usr/bin/env python3
"""script to define a single neurion performing binary classification"""

import numpy as np
import matplotlib.pyplot as plt


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
    
    def MSE(self, Y, A):
        m = Y.shape[1]
        C = 1 / m * np.sum((A - Y) ** 2)
        return C

    def evaluate(self, X, Y):
        """
        evaluates the neuron's predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        predict = np.where(self.__A >= 0.5, 1, 0)
        MSE = self.MSE(Y, self.__A)
        return predict, cost, MSE

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dz = A - Y
        dW = np.matmul(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W -= (alpha * dW).T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        training neuron
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        steps = 0
        a = np.zeros(iterations + 1)
        for i in range(iterations+1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            if (i==steps or i==iterations) and step:
                print("Cost after {} iterations: {}".format(i, cost))
                steps += step
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)
            if graph is True:
                a[i] = cost
        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(np.arrange(0, iterations+1), a)
            plt.show()
        return self.evaluate(X, Y)
