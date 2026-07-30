#!/usr/bin/env python3
"""Single-order n-gram BLEU score without external dependencies.

Implements clipped n-gram precision with a brevity penalty for a specific
order n. Inputs are pre-tokenized lists of strings.
"""
from collections import Counter
import numpy as np


def generate_ngram(sentence, order):
    """Generate contiguous n-grams from a tokenized sentence.

    Args:
        sentence (list[str]): Tokenized sentence.
        order (int): N-gram order (n >= 1).

    Returns:
        list[str]: List of n-grams joined by single spaces.
    """
    ngrams = []
    for i in range(len(sentence) - order + 1):
        ngram = sentence[i: i + order]
        ngrams.append(' '.join(ngram))

    return ngrams


def modified_precision(references, sentence, order):
    """Calculate clipped n-gram precision for a given order.

    Args:
        references (list[list[str]]): Reference translations; tokenized.
        sentence (list[str]): Candidate translation; tokenized.
        order (int): N-gram order.

    Returns:
        float: Clipped precision in [0, 1]. Returns 0 if no n-grams exist.
    """
    sentence_ngrams = Counter(generate_ngram(sentence, order))

    if not sentence_ngrams:
        return 0

    max_counts = {}
    for reference in references:
        ref_ngrams = Counter(generate_ngram(reference, order))
        for ngram in sentence_ngrams:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    ref_ngrams[ngram])

    # Intersection with clipping: limit by max reference count per n-gram
    clipped_counts = {
        ngram: min(count, max_counts.get(ngram, 0))
        for ngram, count in sentence_ngrams.items()
    }

    numerator = sum(clipped_counts.values())
    denominator = max(1, len(sentence) - order + 1)

    return numerator / denominator


def ngram_bleu(references, sentence, n):
    """Compute BLEU for a single n-gram order.

    Args:
        references (list[list[str]]): Reference translations; tokenized.
        sentence (list[str]): Candidate translation; tokenized.
        n (int): N-gram order to evaluate.

    Returns:
        float: BLEU score = brevity_penalty * modified_precision.
    """

    len_sentence = len(sentence)
    ngram_precision = modified_precision(references, sentence, n)

    closest_ref_len = min((abs(len(ref) - len_sentence), len(ref))
                          for ref in references)[1]
    if len_sentence > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / len_sentence)

    bleu_score = BP * ngram_precision

    return bleu_score
