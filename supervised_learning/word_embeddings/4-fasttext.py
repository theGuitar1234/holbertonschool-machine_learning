#!/usr/bin/env python3
"""Module for training a FastText model."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a Gensim FastText model.

    Args:
        sentences (list): List of tokenized sentences to train on.
        vector_size (int): Dimensionality of the word vectors.
        min_count (int): Minimum word frequency required for training.
        negative (int): Size of negative sampling.
        window (int): Maximum distance between current and predicted words.
        cbow (bool): Whether to use CBOW. If False, use Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.FastText: The trained FastText model.
    """
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=not cbow,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
