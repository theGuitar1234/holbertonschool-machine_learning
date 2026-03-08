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

        self.W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.randn(nodes).reshape(1, nodes)
        self.b2 = 0
        self.A2 = 0
