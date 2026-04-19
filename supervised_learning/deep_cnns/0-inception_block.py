#!/usr/bin/env python3
"""Script to create an inception block"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    F1 is the number of filters in the 1x1 convolution

    F3R is the number of filters in the 1x1 convolution before the 3x3

    F3 is the number of filters in the 3x3 convolution

    F5R is the number of filters in the 1x1 convolution before the 5x5

    F5 is the number of filters in the 5x5 convolution

    FPP is the number of filters in the 1x1 convolution after the max pooling

    All convolutions inside the inception block should use a rectified linear
    activation (ReLU)

    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()

    conv_1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    conv_3r = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    conv_3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv_3r)

    conv_5r = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    conv_5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv_5r)

    pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(pool)

    return K.layers.concatenate([conv_1, conv_3, conv_5, pool_proj])
