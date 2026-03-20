#!/usr/bin/env python3
'''
momentum optimization
'''

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    '''
    alpha is the learning rate

    beta1 is the momentum weight

    Returns: momentum optimization operation
    '''

    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
