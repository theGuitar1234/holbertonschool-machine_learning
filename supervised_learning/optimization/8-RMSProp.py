#!/usr/bin/env python3
'''
RMSProp optimization
'''

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    '''
    alpha is the learning rate

    beta2 is the RMSProp weight

    epsilon is a small number to avoid division by zero

    Returns: RMSProp optimization operation
    '''

    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2,
                                       epsilon=epsilon)
