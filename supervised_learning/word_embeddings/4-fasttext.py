#!/usr/bin/env python3
"""
    Train fastText model
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model.

    Args:
        sentences: List of sentences to be trained on.
        vector_size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences of a word for use in training.
        negative: Size of negative sampling.
        window: Maximum distance between the current and predicted word within
            a sentence.
        cbow: Boolean to determine the training type. True is for CBOW,
            False is for Skip-gram.
        epochs: Number of iterations to train over.
        seed: Seed for the random number generator.
        workers: Number of worker threads to train the model.

    Returns:
        The trained model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(sentences=sentences,
                                   sg=sg,
                                   vector_size=vector_size,
                                   negative=negative,
                                   window=window,
                                   min_count=min_count,
                                   seed=seed,
                                   workers=workers)

    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
