#!/usr/bin/env python3
"""Docstring for 1-input.py."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Docstring for build_model."""
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.L2(lambtha)
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
