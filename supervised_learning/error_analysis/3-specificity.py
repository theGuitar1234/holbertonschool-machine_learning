#!/usr/bin/env python3
"""3-specificity.py."""

import numpy as np


def specificity(confusion):
    """Docstring for specificity."""
    total = np.sum(confusion)
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = total - (tp + fp + fn)

    return tn / (tn + fp)
