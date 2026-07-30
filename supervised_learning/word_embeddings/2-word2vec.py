#!/usr/bin/env python3
"""
    Train Word2Vec
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model.

    Args:
        sentences: List of sentences to be trained on.
        vector_size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences of a word for use in training.
        window: Maximum distance between the current and predicted word within
            a sentence.
        negative: Size of negative sampling.
        cbow: Boolean to determine training type. True is for CBOW,
            False is for Skip-gram.
        epochs: Number of iterations to train over.
        seed: Seed for the random number generator.
        workers: Number of worker threads to train the model.

    Returns:
        The trained model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   negative=negative,
                                   seed=seed,
                                   workers=workers,
                                   sg=sg)
    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
