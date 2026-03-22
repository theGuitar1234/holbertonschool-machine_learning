#!/usr/bin/env python3
'''
creating layers
'''

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambd):
    '''
    prev - a tensor containing the output of the previous layer
    n - the number of nodes of the new layer should contain
    activation - the activation function that should be used on the layer
    lambd - the L2 regularization parameter
    '''

    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_avg'
        ),
        kernel_regularizer=tf.keras.regularizers.L2(lambd)
    )
    return layer(prev)
