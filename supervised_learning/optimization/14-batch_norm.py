#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer"""
    active = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=None,
                                  kernel_initializer=active,
                                  name='layer')(prev)
    mu, sigma_2 = tf.nn.moments(layer, axes=[0])
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name='gamma')
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]),
                       name='beta')
    Z_b_norm = tf.nn.batch_normalization(
        layer, mu,
        sigma_2,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-7
    )
    return activation(Z_b_norm)
