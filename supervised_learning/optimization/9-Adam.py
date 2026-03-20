#!/usr/bin/env python3
'''
Adam optimization
'''

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''
    Update a variable using the Adam optimization algorithm

    Parameters:
    alpha: learning rate
    beta1: weight for the first moment
    beta2: weight for the second moment
    epsilon: small number to avoid division by zero
    var: numpy.ndarray containing the variable to be updated
    grad: numpy.ndarray containing the gradient of var
    v: numpy.ndarray containing the moving average of the first moment of grad
    s: numpy.ndarray containing the moving average of the second moment of grad
    t: time step

    Returns:
    Updated variable, new moving average of the first moment,
    new moving average of the second moment
    '''
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first moment estimate
    v_corrected = v / (1 - beta1 ** t)

    # Compute bias-corrected second raw moment estimate
    s_corrected = s / (1 - beta2 ** t)

    # Update variable
    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v, s
