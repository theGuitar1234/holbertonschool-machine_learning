#!/usr/bin/env python3
"""
Bag Of Words
This module implements a bag-of-words embedding function.
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): List of vocabulary words to use for
            the analysis. If None, all words within sentences should be used.

    Returns:
        tuple: (embeddings, features)
            - embeddings (numpy.ndarray): Shape (s, f) containing embeddings.
                s is the number of sentences in sentences.
                f is the number of features analyzed.
            - features (list): List of the features used for embeddings.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
        preprocessed_sentences.append(preprocessed_sentence)

    list_words = []
    for sentence in preprocessed_sentences:
        words = re.findall(r'\w+', sentence)
        list_words.extend(words)

    if vocab is None:
        vocab = sorted(set(list_words))
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    features = np.array(vocab)
    for i, sentence in enumerate(sentences):
        words = re.findall(r'\w+', sentence.lower())
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1
    return embeddings, features
