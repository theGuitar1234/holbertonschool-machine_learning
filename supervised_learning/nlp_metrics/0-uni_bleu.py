#!/usr/bin/env python3
"""BLEU-1 (unigram) score without external dependencies.

This module implements the unigram BLEU score with clipped counts and the
brevity penalty as defined in Papineni et al. (2002). Inputs are tokenized
as lists of strings; no NLTK is required.
"""
from collections import Counter

import numpy as np


def uni_bleu(references, sentence):
    """Compute BLEU-1 for a candidate sentence.

    Args:
        references (list[list[str]]): Reference translations; each is a list
            of tokens.
        sentence (list[str]): Candidate translation as a list of tokens.

    Returns:
        float: BLEU-1 in [0, 1], using clipped unigram precision and brevity
            penalty based on the closest reference length.
    """

    len_sentence = len(sentence)
    sentence_counts = Counter(sentence)

    closest_ref_len = min((abs(len(ref) - len_sentence), len(ref))
                          for ref in references)[1]

    if len_sentence > closest_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - closest_ref_len / len_sentence)

    max_ref_counts = {}
    for ref in references:
        ref_counts = Counter(ref)
        for word in sentence_counts:
            max_ref_counts[word] = max(max_ref_counts.get(word, 0),
                                       ref_counts.get(word, 0))

    modified_precision = sum(min(max_ref_counts.get(word, 0),
                                 sentence_counts[word])
                             for word in sentence_counts) / len_sentence

    bleu_score = BP * modified_precision

    return bleu_score
