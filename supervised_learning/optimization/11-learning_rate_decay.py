#!/usr/bin/env python3
'''
learning rate decay
'''

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''
    Updates the learning rate using inverse time decay in numpy

    Parameters:
    alpha: original learning rate
    decay_rate: weight used to determine the rate at which alpha will decay
    global_step: number of passes of gradient descent that have elapsed
    decay_step: number of passes of gradient descent that should occur
    before alpha is decayed

    Returns:
    Updated learning rate
    '''
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
