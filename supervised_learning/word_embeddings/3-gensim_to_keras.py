#!/usr/bin/env python3
"""Module for converting a Gensim Word2Vec model to Keras."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: Trained Gensim Word2Vec model.

    Returns:
        tf.keras.layers.Embedding: A trainable Keras Embedding layer
        initialized with the Word2Vec vectors.
    """
    weights = model.wv.vectors

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return layer
