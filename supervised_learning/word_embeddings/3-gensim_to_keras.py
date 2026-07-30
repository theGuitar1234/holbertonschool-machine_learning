#!/usr/bin/env python3
"""
    Extract Word2Vec
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Args:
        model: Trained gensim word2vec model.

    Returns:
        trainable keras Embedding layer.
    """
    weights = model.wv.vectors

    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
