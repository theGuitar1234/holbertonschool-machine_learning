#!/usr/bin/env python3
"""Tests something."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """does something."""
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
