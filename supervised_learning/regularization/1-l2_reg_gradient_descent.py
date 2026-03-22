#!/usr/bin/env python3
'''
gradient_descent_cost_with_L2
'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambd, L):
    '''
    Y - a one-hot numpy.ndarray of shape (classes, m) that contains
    the correct labels for the data
        - classes: the number of classes
        - m: the number of data points

    weights - a dictionary of the weights and biases of the neural network
    cache - a dictionary of the outputs of each layer of the neural network
    alpha - the learning rate
    lambd - the L2 regularization parameter
    L - the number of layers of the network
    '''
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambd / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
