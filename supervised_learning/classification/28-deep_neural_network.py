#!/usr/bin/env python3
"""Deep neural network for multiclass classification"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if type(nodes) is not int or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            if i == 0:
                self.__weights["W1"] = (np.random.randn(layers[i], nx) *
                                        np.sqrt(2 / nx))
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1])
                )
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    @property
    def activation(self):
        """Getter for activation"""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the network"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]
            Z = np.matmul(W, A_prev) + b

            if i == self.__L:
                T = np.exp(Z)
                A = T / np.sum(T, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """Evaluates the network's predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the network"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(i - 1)]
            W = weights_copy["W{}".format(i)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(i)] = (
                self.__weights["W{}".format(i)] - alpha * dW
            )
            self.__weights["b{}".format(i)] = (
                self.__weights["b{}".format(i)] - alpha * db
            )

            if i > 1:
                A_curr = cache["A{}".format(i - 1)]
                if self.__activation == 'sig':
                    dZ = np.matmul(W.T, dZ) * (A_curr * (1 - A_curr))
                else:
                    dZ = np.matmul(W.T, dZ) * (1 - A_curr ** 2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i == 0 or i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    costs.append(c)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
