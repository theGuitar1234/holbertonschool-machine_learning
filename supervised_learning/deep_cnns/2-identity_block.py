#!/usr/bin/env python3
'''
identity block
'''

from tensorflow import keras as K


def identity_block(A_prev, filters):
    '''
    A_prev - the output from the previous layer
    filters - a tuple or list containing F11, F3, F12, respectively:
        F11 - the number of filters in the first 1x1 convolution
        F3 - the number of filters in the 3x3 convolution
        F12 - the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels
    axis and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the identity block
    '''

    init = K.initializers.he_normal(seed=0)
    activation = 'relu'
    F11, F3, F12 = filters

    X = K.layers.Conv2D(F11,
                        (1, 1),
                        padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation(activation)(X)
    X = K.layers.Conv2D(F3,
                        (3, 3),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation(activation)(X)
    X = K.layers.Conv2D(F12,
                        (1, 1),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])

    X = K.layers.Activation(activation)(X)

    return X
