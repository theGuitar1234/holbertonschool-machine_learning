#!/usr/bin/env python3
"""2-precision.py."""

import numpy as np


def precision(confusion):
    """Docstring for percision."""
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return true_positives / predicted_positives
