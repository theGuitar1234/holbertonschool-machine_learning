#!/usr/bin/env python3
"""11-config.py."""

import tensorflow.keras as K


def save_config(network, filename):
    """save_config."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(network.to_json())


def load_config(filename):
    """load_config."""
    with open(filename, 'r', encoding='utf-8') as f:
        return K.models.model_from_json(f.read())
