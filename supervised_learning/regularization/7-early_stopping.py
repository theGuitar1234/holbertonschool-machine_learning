#!/usr/bin/env python3
'''
early stopping
'''

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''
    Early stopping should occur when the validation cost of
    the network has not decreased relative
    to the optimal validation cost by more
    than the threshold over a specific patience count

    cost - the current validation cost of the neural network
    opt_cost - the lowest recorded validation cost of the neural network
    threshold - the threshold used for early stopping
    patience - the patience count used for early stopping
    count - the count of how long the threshold has not been met

    Returns: a boolean of whether the network should be stopped early,
    followed by the updated count
    '''

    if opt_cost - cost > threshold:
        return False, 0
    elif opt_cost - cost < threshold:
        count += 1
        if count >= patience:
            return True, count
    return False, count
