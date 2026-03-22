#!/usr/bin/env python3
'''
regularization_cost_with_L2
'''

import numpy as np


def l2_reg_cost(cost, lambd, weights, L, m):
    """
    cost - the cost of the network without L2 regularization
    lambd - the regularization parameter
    weights - a dict of the weights and biases(ndarray) of the neural network
    L - the number of layers in the neural network
    m - the number of data points used
    """
    sq_weights = 0
    for i in range(1, L+1):
        sq_weights += np.sum(np.square(weights[f"W{i}"]))

    l2_penalty = (lambd / (2 * m)) * sq_weights
    return cost + l2_penalty
