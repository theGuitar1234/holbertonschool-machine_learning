#!/usr/bin/env python3
"""
    TF-IDF
"""
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.

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

    if len(sentences) == 0:
        return np.array([]), []

    preprocessed_sentences = []
    for sentence in sentences:
        preprocessed_sentence = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
        preprocessed_sentences.append(preprocessed_sentence)

    if vocab is None:
        all_words = []
        for sentence in preprocessed_sentences:
            words = re.findall(r'\w+', sentence)
            all_words.extend(words)
        vocab = sorted(set(all_words))
    else:
        # Convert vocab to lowercase to match preprocessed sentences
        vocab = [word.lower() for word in vocab]

    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        lowercase=False,  # Already lowercase
        token_pattern=r'\b\w+\b',  # Match word boundaries
        smooth_idf=True,  # Add 1 to idf (default)
        norm='l2'  # L2 normalization (default)
    )

    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    features = np.array(vectorizer.get_feature_names_out())

    return tfidf_matrix.toarray(), features
