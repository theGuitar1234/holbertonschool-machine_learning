#!/usr/bin/env python3
"""Docstring for 3-one_hot.py."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Docstring for one_hot."""
    return K.utils.to_categorical(labels, num_classes=classes)
