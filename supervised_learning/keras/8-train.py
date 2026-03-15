#!/usr/bin/env python3
"""8-train.py."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """train_model."""
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(
                K.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience
                )
            )

        if learning_rate_decay:
            def scheduler(epoch):
                """scheduler."""
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(
                K.callbacks.LearningRateScheduler(
                    scheduler,
                    verbose=1
                )
            )

        if save_best:
            callbacks.append(
                K.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            )

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
