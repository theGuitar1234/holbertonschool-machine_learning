#!/usr/bin/env python3
"""9-model.py."""

import tensorflow.keras as K


def save_model(network, filename):
    """save_model."""
    network.save(filename)


def load_model(filename):
    """load_model."""
    return K.models.load_model(filename)
