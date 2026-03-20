#!/usr/bin/env python3
'''
learning rate decay upgraded
'''

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    '''
    Updates the learning rate using inverse time decay in tensorflow

    Parameters:
    alpha: original learning rate
    decay_rate: weight used to determine the rate at which alpha will decay
    decay_step: number of passes of gradient descent that should occur
    before alpha is decayed

    Returns:
    Updated learning rate
    '''
    return tf.keras.optimizers.schedules.InverseTimeDecay(
            alpha,
            decay_step,
            decay_rate,
            staircase=True)
