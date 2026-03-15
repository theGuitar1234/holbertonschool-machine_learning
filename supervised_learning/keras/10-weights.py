#!/usr/bin/env python3
"""10-weights.py."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """save_weights."""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """load_weights."""
    network.load_weights(filename)
