#!/usr/bin/env python3
"""4-f1_score.py."""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Docstring for f1_score."""
    rec = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * rec) / (prec + rec)
