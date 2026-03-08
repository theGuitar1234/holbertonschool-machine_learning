#!/usr/bin/env python3
"""script to create neural network"""

import numpy as np


class NeuralNetwork:
    """
    class for neuralnetwork
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Weight vector 1 hidden layer

        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Bias1

        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Activated1

        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Weight vector 2

        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Bias2

        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Activated output 2 prediction

        """
        return self.__A2

    def forward_prop(self, X):
        """
        propagation of neural network
        """
        E1 = np.matmul(self.__W1, X) + self.__b1
        sigmoid1 = 1 / (1 + np.exp(-E1))
        self.__A1 = sigmoid1
        E2 = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid2 = 1 / (1 + np.exp(-E2))
        self.__A2 = sigmoid2
        return self.__A1, self.__A2

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

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        predict = np.where(self.__A2 >= 0.5, 1, 0)
        return predict, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        returning gradient_descent
        """

        m = Y.shape[1]
        dz2 = A2 - Y  # derivative z2
        dW2 = np.matmul(A1, dz2.T) / m  # grad of the loss with respect to w

        # grad of the loss with respect to b
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dW1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W2 -= (alpha * dW2).T
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

        def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
            """
            training network
            """
            if type(iterations) is not int:
                raise TypeError("iterations must be an integer")
            if iterations < 0:
                raise ValueError("iterations must be a positive integer")
            if type(alpha) is not float:
                raise TypeError("alpha must be a float")
            if alpha < 0:
                raise ValueError("alpha must be positive")

            steps = 0
            c_ax = np.zeros(iterations + 1)

            temp_cost = []
            temp_iterations = []
            for i in range(iterations + 1):
                self.forward_prop(X)
                cost = self.cost(Y, self.__A2)
                if i % step == 0 or i == iterations:
                    temp_cost.append(cost)
                    temp_iterations.append(i)
                    if verbose is True:
                        print("Cost after {} iterations: {}".format(i, cost))
                if i < iterations:
                    self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)


            if graph is True:
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
                plt.plot(temp_iterations, temp_cost)
                plt.show()
            return self.evaluate(X, Y)
