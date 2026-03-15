#!/usr/bin/env python3
"""6-train.py."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """train_model."""
    callbacks = []

    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
