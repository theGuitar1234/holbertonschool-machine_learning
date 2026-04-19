#!/usr/bin/env python3
'''
dense blocks
'''

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    '''
    X - the output from the previous layer
    nb_filters - an integer representing the number of filters in X
    growth_rate - the growth rate for the dense block
    layers - the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively

    Returns: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    '''

    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):

        batch1 = K.layers.BatchNormalization()(X)

        relu1 = K.layers.Activation('relu')(batch1)

        bottleneck = K.layers.Conv2D(filters=4*growth_rate,
                                     kernel_size=1, padding='same',
                                     kernel_initializer=init)(relu1)

        batch2 = K.layers.BatchNormalization()(bottleneck)

        relu2 = K.layers.Activation('relu')(batch2)

        X_conv = K.layers.Conv2D(filters=growth_rate,
                                 kernel_size=3, padding='same',
                                 kernel_initializer=init)(relu2)

        X = K.layers.concatenate([X, X_conv])
        nb_filters += growth_rate

    return X, nb_filters
