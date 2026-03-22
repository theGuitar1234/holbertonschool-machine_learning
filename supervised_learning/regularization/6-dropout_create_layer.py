#!/usr/bin/env python3
'''
create a new layer with dropout
'''

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    '''
    prev - a tensor containing the output of the previous layer
    n - the number of nodes the new layer should contain
    activation - the activation function for the new layer
    keep_prob - the probability that a node will be kept
    training - a boolean indicating whether the model is in training mode

    Returns: the output of the new layer
    '''
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)
    output = layer(prev)
    output = tf.keras.layers.Dropout(rate=1-keep_prob)(output,
                                                       training=training)
    return output
