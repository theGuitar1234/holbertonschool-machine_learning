#!/usr/bin/env python3
'''
projection block
'''

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    '''
    A_prev - the output from the previous layer
    filters - a tuple or list containing F11, F3, F12, respectively:
        F11 - the number of filters in the first 1x1 convolution
        F3 - the number of filters in the 3x3 convolution
        F12 - the number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection
    s - the stride of the first convolution in both the main path
        and the shortcut connection
    All convolutions inside the block should be followed by
    batch normalization along the channels axis and a rectified
    linear activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the projection block
    '''

    init = K.initializers.he_normal(seed=0)
    activation = 'relu'
    F11, F3, F12 = filters

    conv1 = K.layers.Conv2D(F11, kernel_size=1, strides=s,
                            padding='same', kernel_initializer=init)(A_prev)

    batch1 = K.layers.BatchNormalization(axis=3)(conv1)

    relu1 = K.layers.Activation(activation)(batch1)

    conv2 = K.layers.Conv2D(F3, kernel_size=3, strides=1,
                            padding='same', kernel_initializer=init)(relu1)

    batch2 = K.layers.BatchNormalization(axis=3)(conv2)

    relu2 = K.layers.Activation(activation)(batch2)

    conv3 = K.layers.Conv2D(F12, kernel_size=1, strides=1,
                            padding='same', kernel_initializer=init)(relu2)

    conv1_proj = K.layers.Conv2D(F12, kernel_size=1, strides=s,
                                 padding='same',
                                 kernel_initializer=init)(A_prev)

    batch3 = K.layers.BatchNormalization(axis=3)(conv3)
    batch4 = K.layers.BatchNormalization(axis=3)(conv1_proj)

    add = K.layers.Add()([batch3, batch4])

    final_relu = K.layers.Activation(activation)(add)

    return final_relu
