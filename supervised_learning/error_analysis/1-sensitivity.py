#!/usr/bin/env python3
"""1-sensitivity.py."""

import numpy as np


def sensitivity(confusion):
    """Docstring for sensitivity."""
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)  # TP + FN for each class
    return true_positives / actual_positives
