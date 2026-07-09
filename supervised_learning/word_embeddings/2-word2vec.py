#!/usr/bin/env python3
"""Module for training a Word2Vec model."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a Gensim Word2Vec model.

    Args:
        sentences (list): List of tokenized sentences to train on.
        vector_size (int): Dimensionality of the word vectors.
        min_count (int): Minimum word frequency required for training.
        window (int): Maximum distance between current and predicted words.
        negative (int): Size of negative sampling.
        cbow (bool): Whether to use CBOW. If False, use Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
