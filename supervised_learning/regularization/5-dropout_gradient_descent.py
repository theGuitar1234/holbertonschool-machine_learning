#!/usr/bin/env python3
'''
Dropout gradient descent
'''

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''
    Y - a one-hot numpy.ndarray of shape (classes, m) containing the correct
        labels for the data
        - classes is the number of classes
        - m is the number of data points
    weights - a dictionary of the weights and biases of the neural network
    cache - a dictionary of the outputs and dropout masks of each layer of
            the neural network
    alpha - the learning rate
    keep_prob - the probability that a node will be kept
    L - the number of layers in the network
    '''
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            D = cache['D' + str(i - 1)]
            dA_prev *= D
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
